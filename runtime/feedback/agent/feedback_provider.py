"""One bounded provider call, registered with the native Omega provider API."""
import json
import os
from pathlib import Path
import time

import openai
import providers
from runtime.omega import llm_config as config
from runtime.omega.llm_gateway import completion_gateway
from reason_schema import schema_for


class FeedbackProvider(providers.LLMProvider):
    def start(self):
        config.validate()
        self.calls = 0
        self.client = openai.OpenAI(base_url=config.base_url(), api_key=os.environ['LLM_API_KEY'],
                                  timeout=float(os.environ['FEEDBACK_REQUEST_TIMEOUT']), max_retries=0)

    def chat(self, prompt, max_tokens=2048, reasoning_mode='none'):
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError('Feedback model-call budget exhausted; no automatic retry.')
        started = time.monotonic()
        try:
            call = completion_gateway(self.client.chat.completions.create)
            schema=schema_for(json.loads(Path('context.json').read_text()))
            response = call(model=os.environ['LLM_MODEL'], messages=[
                {'role':'system','content':str(prompt)}], max_tokens=int(max_tokens), temperature=0,
                response_format={'type':'json_schema','json_schema':{'name':'feedback_review','strict':True,'schema':schema}})
            choice = response.choices[0]
            if choice.finish_reason == 'length':
                raise ValueError('Omega response was truncated; no further model request was made.')
            result = json.loads(choice.message.content)
            Path('model-proposal.json').write_text(json.dumps(result))
            usage = getattr(response, 'usage', None)
            metadata = {'model':os.environ['LLM_MODEL'], 'provider':config.provider(), 'model_calls':1,
                        'prompt_tokens':getattr(usage,'prompt_tokens',None),
                        'completion_tokens':getattr(usage,'completion_tokens',None),
                        'seconds':round(time.monotonic()-started,2)}
            Path('review-usage.json').write_text(json.dumps(metadata))
            return 'finish-review '+json.dumps(result,separators=(',',':'))
        except Exception as error:
            status = getattr(error,'status_code',None)
            message = ('Omega provider rejected the request (HTTP '+str(status)+').' if status else
                       'Omega review interpretation failed ('+type(error).__name__+').')
            if isinstance(error,ValueError):
                message = str(error)[:250]
            Path('failure.json').write_text(json.dumps({'error':message,'model_calls':self.calls,'retryable':False}))
            os._exit(2)


def loadOmegaPlugin():
    providers.registerLLMProvider('Feedback',FeedbackProvider())
