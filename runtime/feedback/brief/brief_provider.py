"""One bounded configured-model request through native Omega's provider API."""
import json
import os
from pathlib import Path
import time
import openai
import providers
from runtime.omega import llm_config as config
from runtime.omega.llm_gateway import completion_gateway
from brief_schema import schema_for


class BriefProvider(providers.LLMProvider):
    def start(self):
        config.validate();self.calls=0
        self.client=openai.OpenAI(base_url=config.base_url(),api_key=os.environ['LLM_API_KEY'],
                                 timeout=float(os.environ['FEEDBACK_REQUEST_TIMEOUT']),max_retries=0)

    def chat(self,prompt,max_tokens=1800,reasoning_mode='none'):
        self.calls+=1
        if self.calls>1:raise RuntimeError('Optional-brief request budget exhausted; no automatic retry.')
        start=time.monotonic()
        try:
            call=completion_gateway(self.client.chat.completions.create)
            response=call(model=os.environ['LLM_MODEL'],messages=[{'role':'system','content':str(prompt)}],
                          max_tokens=int(max_tokens),temperature=0,response_format={'type':'json_schema','json_schema':{
                              'name':'readonly_review_brief','strict':True,'schema':schema_for(json.loads(Path('context.json').read_text()))}})
            choice=response.choices[0]
            if choice.finish_reason=='length':raise ValueError('The optional brief was truncated. No automatic retry was made.')
            value=json.loads(choice.message.content)
            Path('model-proposal.json').write_text(json.dumps(value))
            usage=getattr(response,'usage',None)
            Path('review-usage.json').write_text(json.dumps(dict(model=os.environ['LLM_MODEL'],provider=config.provider(),model_calls=1,
                prompt_tokens=getattr(usage,'prompt_tokens',None),completion_tokens=getattr(usage,'completion_tokens',None),
                seconds=round(time.monotonic()-start,2))))
            return 'finish-brief '+json.dumps(value,separators=(',',':'))
        except Exception as error:
            code=getattr(error,'status_code',None)
            message='The optional Omega provider request failed'+(' (HTTP '+str(code)+').' if code else ' ('+type(error).__name__+').')
            if isinstance(error,ValueError):message=str(error)[:250]
            Path('failure.json').write_text(json.dumps({'error':message,'model_calls':self.calls,'retryable':False}));os._exit(2)


def loadOmegaPlugin():providers.registerLLMProvider('Brief',BriefProvider())
