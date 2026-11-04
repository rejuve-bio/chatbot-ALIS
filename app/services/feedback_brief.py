"""Optional Omega explanations, isolated from the review/PLN/atlas write paths."""
from __future__ import annotations

import copy
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from app.services.feedback_brief_schema import POLICY,digest,validate
from app.services.feedback_omega import OmegaReviewClient,OmegaReviewError,redact
from app.services.feedback_triage import review_status

ROOT=Path(__file__).resolve().parents[2]
PROFILE=ROOT/'runtime/feedback/brief'


def packet(store,candidate,sources):
    reviews=store.current_reviews(candidate);a=store.belief_assessment(candidate)
    # A bounded, declared sample of notes, balanced by the selected judgment.
    selected=[]
    for decision in ('support','oppose','uncertain'):
        selected.extend([r for r in reviews if r['decision']==decision and (r['rationale'].strip() or r['concern_text'].strip())][-2:])
    identities=[candidate['patient_id']]
    for r in reviews:identities.extend([r['clinician_id'],r.get('clinician_email'),r['reviewer']])
    request=dict(policy=POLICY,claim=candidate['proposition'],intervention=candidate['name'],condition=candidate['condition'],
                 counts={k:(a or {}).get(k,0) for k in ('support','oppose','uncertain')},
                 status=review_status(a,reviews)['label'],sources=[],review_notes=[],
                 coverage={'current_reviews':len(reviews),'reviews_with_notes':sum(bool(r['rationale'].strip() or r['concern_text'].strip()) for r in reviews),
                           'selected_notes':len(selected),'selection':'Up to two latest nonempty notes per judgment category.',
                           'sources_inspected':0})
    for source in sources[:2]:
        if not source or not source.get('abstract_available'):continue
        if source['id'] not in store.claim_source_ids(candidate):raise ValueError('Brief source is not linked to this intervention.')
        source_text=source.get('text','')
        if hashlib.sha256(source_text.encode()).hexdigest()!=source.get('text_sha256'):raise ValueError('Brief source failed its saved-text check.')
        request['sources'].append({'id':source['id'],'text':source_text[:7000],'truncated':len(source_text)>7000})
    for r in selected:
        # Notes remain opinions. Even a reviewer-claimed correction is not made a source fact.
        value='Decision: '+r['decision']+'\nNote: '+r['rationale'][:700]
        if r['concern_kind']:value+='\nFlag ('+r['concern_kind']+'): '+r['concern_text'][:500]
        request['review_notes'].append({'id':'Review:#'+str(r['id']),'text':value})
    request['coverage']['sources_inspected']=len(request['sources'])
    return redact(request,identities)


class OmegaBriefClient(OmegaReviewClient):
    def __init__(self,container=None,timeout=45,output_tokens=1800):
        super().__init__(container=container,timeout=timeout,output_tokens=output_tokens)
        self.files={name:(PROFILE/name).read_text() for name in ('run.metta','skills.metta','channels.metta','memory.metta','bridge.py','brief_provider.py')}
        self.files['brief_schema.py']=(ROOT/'app/services/feedback_brief_schema.py').read_text()
        self.profile_hash=hashlib.sha256(json.dumps(self.files,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def identity(self):
        try:return super().identity()
        except OmegaReviewError:
            raise OmegaReviewError('The optional Omega explanation service is unavailable or unconfigured. Reviewing, native PLN updates and saved atlas data are still available.') from None

    def brief_key(self,request):return digest({'policy':POLICY,'runtime':self.identity(),'request':request})

    def explain(self,request,progress=None):
        identity=self.identity()
        if len(json.dumps(request).encode())>21000:raise OmegaReviewError('The explanation packet exceeds the bounded input limit. No model call was made.')
        package=dict(request=request,files=self.files,expected_core=self.manifest['omega_commit'],expected_petta=self.manifest['petta_commit'],
                     output_tokens=self.output_tokens,request_timeout=self.timeout)
        command=['docker','exec','-i',self.container,'python3','-B','-c',(PROFILE/'runner.py').read_text()]
        process=None
        try:
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            first=True;started=time.monotonic()
            while True:
                try:
                    stdout,stderr=process.communicate(json.dumps(package) if first else None,timeout=15);break
                except subprocess.TimeoutExpired:
                    first=False
                    if time.monotonic()-started>self.timeout+35:
                        process.kill();process.communicate()
                        raise OmegaReviewError('The optional Omega brief timed out. Reviews and scores are unchanged; no automatic retry was made.')
                    if progress:progress('Still waiting for the same optional Omega request; no additional model call.')
            result=json.loads(stdout)
            if process.returncode or result.get('error'):raise OmegaReviewError(result.get('error','Omega did not finish the optional review brief.'),diagnostic=result)
            if (result['policy']!=POLICY or result['request_digest']!=digest(request) or result['context']!=request
                    or result['usage']['model_calls']!=1 or result['execution']!='native-omega-core-tool-loop'
                    or any(result['runtime'][k]!=identity[k] for k in ('core_commit','petta_commit','model','profile_sha256'))
                    or [x['tool'] for x in result['tool_trace']]!=['read-brief-context','finish-brief']):
                raise ValueError('Brief provenance mismatch.')
            validate(result['brief'],request)
            return result
        except KeyboardInterrupt:
            if process and process.poll() is None:process.kill();process.communicate()
            raise
        except (OSError,ValueError,KeyError,TypeError) as error:
            raise OmegaReviewError('The optional Omega brief could not be verified. Your reviews and scores are unchanged. No automatic retry was made.') from error


def reported_number_filter(value,context):
    """Omit unreported calculated quantities from a draft; never repair them by guessing.

    The original model output stays in the integrity-checked cache for audit.
    This affects only the optional explanation, never a review or truth value.
    """
    from app.services.feedback_brief_schema import references
    pattern=re.compile(r'(?<![A-Za-z0-9])[-+]?(?:\d+\.\d+|\.\d+|\d+)(?:[eE][-+]?\d+)?')
    def numbers(text):
        result=set()
        for value in pattern.findall(text):
            try:result.add(Decimal(value))
            except InvalidOperation:result.add(value)
        return result
    refs=references(context);brief=copy.deepcopy(value);omissions=[]
    for field in ('study_findings','review_discussion','next_questions'):
        accepted=[]
        for item in brief[field]:
            ids=item['reference_ids'] if field=='next_questions' else [c['id'] for c in item['citations']]
            reported=numbers(' '.join(refs[i] for i in ids))
            text=item['question'] if field=='next_questions' else item['explanation']
            if numbers(text)-reported:
                omissions.append({'section':field,'reason':'An explanation introduced a quantity not reported in its cited text.'})
            else:accepted.append(item)
        brief[field]=accepted
    return brief,omissions


class ReviewBrief:
    """Only cache writes are allowed; no review, source correction or rank writes."""
    def __init__(self,session,client=None):self.s=session;self.client=client or OmegaBriefClient()

    def context(self,candidate,sources):
        authorized_dataset=self.s.store.active_dataset()
        self.s.require_access(candidate['patient_id'])
        with self.s.store.transaction():
            if self.s.store.active_dataset()!=authorized_dataset:raise ValueError('The opened handoff changed. Reopen the selected entry.')
            self.s.store.require_active_candidate(candidate['id'])
            current=self.s.store.candidate(candidate['id'])
            if current['current_version']!=candidate['current_version']:raise ValueError('The evidence changed. Reopen the entry before requesting its brief.')
            return packet(self.s.store,current,sources)

    def key(self,candidate,request):
        return digest([POLICY,self.s.store.active_dataset(),candidate['id'],candidate['current_version'],self.client.brief_key(request)])

    def prepare(self,stored):
        if stored is None:return None
        result=copy.deepcopy(stored)
        validate(result['result']['brief'],result['result']['context'])
        brief,omissions=reported_number_filter(result['result']['brief'],result['result']['context'])
        result['result']['brief']=brief;result['result']['omissions']=omissions
        return result

    def cached(self,candidate,request):
        return self.prepare(self.s.store.cached_analysis(self.key(candidate,request),ttl=10**12))

    def generate(self,candidate,sources,request,progress=None):
        current=self.context(candidate,sources)
        if digest(current)!=digest(request):raise ValueError('Reviews changed while preparing the brief. Reopen it to include the latest responses.')
        key=self.key(candidate,request);saved=self.s.store.cached_analysis(key,ttl=10**12)
        if saved:return self.prepare(saved)
        identity=self.client.identity()
        if not self.s.store.has_external_consent(self.s.reviewer,identity):raise ValueError('External-text authorization is required before requesting a model explanation.')
        result=self.client.explain(request,progress=progress)
        if digest(self.context(candidate,sources))!=digest(request):raise ValueError('Reviews changed during explanation. The old brief was not saved as current; review records are unchanged.')
        validate(result['brief'],request)
        return self.prepare(self.s.store.save_analysis(key,result))
