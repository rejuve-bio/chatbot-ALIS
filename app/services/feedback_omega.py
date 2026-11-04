"""Feedback client for the actual native Omega loop in its existing container."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import re
import subprocess
import time

from app.services.feedback_reason_schema import POLICY, digest, validate, apply_reason_policy, validate_review_reason

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / 'runtime/feedback/agent'


class OmegaReviewError(RuntimeError):
    def __init__(self,message,diagnostic=None,*,cancelled=False):
        super().__init__(message)
        self.diagnostic=diagnostic
        self.cancelled=cancelled


def redact(value, identities=()):
    if isinstance(value, dict):
        return {k:redact(v,identities) for k,v in value.items()}
    if isinstance(value, list):
        return [redact(v,identities) for v in value]
    if isinstance(value, str):
        for identity in identities:
            if identity:
                value=value.replace(identity,'[identifier omitted]')
        value=re.sub(r'\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b','[identifier omitted]',value)
        return re.sub(r'[^\s<>@]+@[^\s<>@]+\.[A-Za-z]{2,}', '[email omitted]',value)
    return value


def packet(candidate, decision, reason, citation='', *, clinician=None, prior_reviews=()):
    validate_review_reason(reason)
    data=candidate['data']
    papers=[]
    seen=set()
    # Explicit public PMID references are looked up first, never arbitrary URLs.
    ids=re.findall(r'\bPMID\s*:?\s*(\d{1,10})\b',citation,flags=re.IGNORECASE)
    supplied=list(data.get('publications') or [])
    by_id={str(p.get('pmid','')):p for p in supplied}
    records=[by_id.get(p,{'pmid':p,'title':''}) for p in ids]+supplied
    for p in records:
        pmid=str(p.get('pmid',''))
        if re.fullmatch(r'\d{1,10}',pmid) and pmid not in seen:
            papers.append({'pmid':pmid,'title':str(p.get('title') or '')[:400]})
            seen.add(pmid)
    result={'claim':candidate['proposition'], 'intervention':candidate['name'], 'condition':candidate['condition'],
        'evidence':{'drugs':[str(x)[:120] for x in (data.get('drugs') or [])[:12]],
                    'why':str(data.get('why') or '')[:2400], 'note':str(data.get('note') or '')[:1200],
                    'warning':str(data.get('warning') or data.get('safety_warning') or '')[:1200],
                    'publications':papers[:2]},
        'review':{'decision':decision,'reason':reason,'citation':citation},
        # Other votes do not change the meaning of this reason, invalidate its
        # cache, or become additional experimental evidence for the model.
        'prior_reviews':[]}
    identities=[candidate.get('patient_id')]
    if clinician:
        identities.extend([clinician.id,clinician.subject,clinician.email])
    return redact(result,identities)


class OmegaReviewClient:
    def __init__(self, container=None, timeout=90, output_tokens=4096):
        self.container=container or os.getenv('OMEGA_FEEDBACK_CONTAINER','chatbot-alis-omega-mapping-1')
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}',self.container):
            raise ValueError('Invalid Omega container name.')
        self.timeout=int(timeout)
        self.output_tokens=int(output_tokens)
        if not 10<=self.timeout<=180 or not 512<=self.output_tokens<=4096:
            raise ValueError('Use a feedback timeout of 10–180 seconds and output limit of 512–4096 tokens.')
        names=('run.metta','skills.metta','channels.metta','memory.metta','bridge.py','feedback_provider.py')
        self.files={name:(PROFILE/name).read_text() for name in names}
        self.files['reason_schema.py']=(ROOT/'app/services/feedback_reason_schema.py').read_text()
        self.files['source_schema.py']=(ROOT/'app/services/feedback_source_schema.py').read_text()
        self.profile_hash=hashlib.sha256(json.dumps(self.files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.manifest=json.loads((ROOT/'runtime/feedback/manifest.json').read_text())
        self._identity=None

    def identity(self):
        if self._identity is None:
            program="import os,json; from pathlib import Path; from urllib.parse import urlsplit; m=json.loads(Path('/opt/runtime/manifest.json').read_text()); print(json.dumps({'core_commit':m['core_commit'],'petta_commit':m['petta_commit'],'model':os.getenv('LLM_MODEL'),'provider':os.getenv('LLM_PROVIDER'),'destination':urlsplit(os.getenv('LLM_HOST','')).hostname,'key_configured':bool(os.getenv('LLM_API_KEY'))}))"
            try:
                p=subprocess.run(['docker','exec',self.container,'python3','-B','-c',program],capture_output=True,text=True,timeout=10)
                value=json.loads(p.stdout)
                if p.returncode or not value.get('key_configured') or not value.get('destination') or not value.get('model'):
                    raise ValueError
                if value['core_commit']!=self.manifest['omega_commit'] or value['petta_commit']!=self.manifest['petta_commit']:
                    raise ValueError
            except (OSError,ValueError,KeyError,subprocess.TimeoutExpired):
                raise OmegaReviewError('The native Omega container is unavailable or incompatible. Start '+self.container+' or use --omega off for an explicitly manual review.') from None
            value.pop('key_configured')
            value['profile_sha256']=self.profile_hash
            self._identity=value
        return self._identity

    def cache_key(self, request):
        return digest({'policy':POLICY,'runtime':self.identity(),'request':request})

    def analyze(self, request, progress=None):
        validate_review_reason(request['review']['reason'])
        identity=self.identity()
        package={'request':request,'files':self.files,'expected_core':self.manifest['omega_commit'],
                 'expected_petta':self.manifest['petta_commit'],'output_tokens':self.output_tokens,'request_timeout':self.timeout}
        if len(json.dumps(request).encode())>14000:
            raise OmegaReviewError('This review reason is too long for the bounded Omega packet. Shorten it; no model call was made.')
        command=['docker','exec','-i',self.container,'python3','-B','-c',(PROFILE/'runner.py').read_text()]
        try:
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            encoded=json.dumps(package)
            started=time.monotonic()
            next_notice=15
            first=True
            while True:
                try:
                    stdout,stderr=process.communicate(encoded if first else None,timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    first=False
                    if time.monotonic()-started>self.timeout+50:
                        process.kill();process.communicate()
                        raise OmegaReviewError('Omega review deadline reached. Your draft is saved; no automatic retry was made.')
                    elapsed=int(time.monotonic()-started)
                    if progress and elapsed>=next_notice:
                        progress(f'Still waiting for the same Omega request ({elapsed}s); no additional model call started.')
                        next_notice=elapsed+30
            result=json.loads(stdout)
            if process.returncode or 'error' in result:
                raise OmegaReviewError(str(result.get('error','Native Omega did not finish the review.'))[:500],result)
            if (result['policy']!=POLICY or result['execution']!='native-omega-core-tool-loop' or result['llm_used'] is not True
                    or result['request_digest']!=digest(request)
                    or any(result['runtime'][key]!=identity[key] for key in ('core_commit','petta_commit','model','profile_sha256'))
                    or [t['tool'] for t in result['tool_trace']]!=['read-review-context','finish-review']):
                raise ValueError('Native Omega provenance mismatch.')
            context=dict(result['context']); sources=context.pop('sources')
            if context!=request:
                raise ValueError('Omega changed the review packet.')
            for source in sources:
                if hashlib.sha256(source['text'].encode()).hexdigest()!=source['text_sha256']:
                    raise ValueError('Inspected source hash mismatch.')
            normalized,adjustments=apply_reason_policy(result['raw_analysis'])
            if normalized!=result['analysis'] or adjustments!=result['policy_adjustments']:
                raise ValueError('Reason-policy trace mismatch.')
            validate(result['analysis'],result['context'])
            return result
        except OmegaReviewError:
            raise
        except (OSError,ValueError,KeyError,IndexError,TypeError):
            raise OmegaReviewError('Omega result could not be verified. Your draft is saved; no interpretation or replacement score was accepted.') from None
