"""Source-driven revisions with the actual pinned native PLN interpreter."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.services.feedback_auth import register_local,login_local
from app.services.feedback_omega import packet
from app.services.feedback_reason_schema import validate
from app.services.feedback_source_scores import score_sources
from app.services.feedback_pln import NativePLNError
from app.services.feedback_store import FeedbackStore

# Public reported tuple, used with a deliberately scoped research question.
QUOTE='After adjusting for baseline characteristics, addition of metformin increased the rate of remission (HR 1.39, 95% CI 1.16-1.67, p < 0.001).'
CONTEXT='The primary outcome was prediabetes remission rate. NCT03441750.'


def record():
    text=CONTEXT+' '+QUOTE
    return {'status':'active','source':{'id':'PMID:42655937','text':text,'abstract_available':True,
        'text_sha256':hashlib.sha256(text.encode()).hexdigest()},'facts':{
        'source_id':'PMID:42655937','study_id':'NCT03441750','population_type':'human','design':'randomized',
        'comparison':'target_vs_control','condition_match':'direct','intervention_match':'administered_target','outcome_role':'primary',
        'population':'Adults with impaired glucose regulation','comparator':'Lifestyle intervention alone','outcome':'Prediabetes remission rate',
        'source_quotes':[CONTEXT],'exclusion_reason':'',
        'effect':{'measure':'HR','estimate':'1.39','ci_lower':'1.16','ci_upper':'1.67','ci_level':'95','direction':'higher','quote':QUOTE}}}


class SourceCalculation(unittest.TestCase):
    def test_real_tuple_uses_native_and_repeated_source_adds_no_weight(self):
        r=record();a=score_sources([r]);b=score_sources([r,copy.deepcopy(r)])
        self.assertEqual(a['status'],'calculated')
        self.assertGreater(a['score_100'],80)
        self.assertEqual(a['score_100'],b['score_100'])
        self.assertEqual(a['confidence'],b['confidence'])
        self.assertIn('Truth__w2c',a['native']['program'])
        self.assertNotIn('review_',a['native']['program'])
        self.assertEqual(a['native']['exit_code'],0)

    def test_correcting_direction_changes_score_not_information(self):
        r=record();a=score_sources([r]);r['facts']['effect']['direction']='lower';b=score_sources([r])
        self.assertLess(b['score_100'],a['score_100'])
        self.assertEqual(a['confidence'],b['confidence'])
        self.assertAlmostEqual(a['strength']+b['strength'],1)

    def test_false_ci_and_preclinical_or_incompatible_context_remain_unscored(self):
        for key,value in [('population_type','animal'),('condition_match','related'),('comparison','dose_comparison'),('outcome_role','secondary')]:
            r=record();r['facts'][key]=value
            self.assertIsNone(score_sources([r])['score_100'])
        r=record();r['facts']['effect']['ci_upper']='1.99'
        self.assertIsNone(score_sources([r])['score_100'])
        a=record();b=record();b['facts']['source_id']='PMID:2';b['source']['id']='PMID:2';b['facts']['outcome']='Another endpoint'
        self.assertEqual(score_sources([a,b])['status'],'needs_target')

    def test_null_spanning_ci_is_not_dropped(self):
        r=record();q='The primary mean difference 0.2 (95% CI -0.8 to 1.2).'
        r['source']['text']=q;r['source']['text_sha256']=hashlib.sha256(q.encode()).hexdigest();r['facts']['effect']={'measure':'MD','estimate':'0.2','ci_lower':'-0.8','ci_upper':'1.2','ci_level':'95','direction':'higher','quote':q}
        a=score_sources([r]);self.assertEqual(a['status'],'calculated');self.assertLess(a['confidence'],.1)
        r['facts']['effect']['estimate']='0.0';r['facts']['effect']['ci_lower']='-1.0';r['facts']['effect']['ci_upper']='1.0'
        r['facts']['effect']['quote']='Mean difference 0.0 (95% CI -1.0 to 1.0).'
        r['source']['text']=r['facts']['effect']['quote'];r['source']['text_sha256']=hashlib.sha256(r['source']['text'].encode()).hexdigest()
        a=score_sources([r]);self.assertEqual(a['strength'],.5);self.assertEqual(a['confidence'],0)

    def test_compatible_independent_registered_studies_use_revision(self):
        a=record();b=record();b['source']['id']='PMID:2';b['facts']['source_id']='PMID:2';b['facts']['study_id']='NCT00000002'
        self.assertEqual(score_sources([a,b])['status'],'needs_target')
        a['independence_confirmed']=True;b['independence_confirmed']=True
        result=score_sources([a,b])
        self.assertEqual(result['numerical_input_count'],2)
        self.assertIn('Truth__Revision',result['native']['program'])
        self.assertGreater(result['confidence'],score_sources([a])['confidence'])


class SourceFeedbackLifecycle(unittest.TestCase):
    def test_correction_revises_native_rank_while_opinions_do_not_and_rollback_is_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reviews.sqlite3';store=FeedbackStore(path)
            for name in ('alice','bob'):register_local(store,name,name+'@example.test','Synthetic-pass-only')
            alice=login_local(store,'alice@example.test','Synthetic-pass-only').clinician
            bob=login_local(store,'bob@example.test','Synthetic-pass-only').clinician
            data={'patient_id':'synthetic','risk_areas':[{'disease_name':'Prediabetes remission','investigational_evidence':[
                {'name':'Metformin','drugs':['Metformin'],'why':'Public trial result','publications':[{'pmid':'42655937','title':'China DPP study'}]},
                {'name':'No source','why':'Synthetic missing input'}]}]}
            store.import_data(data,'synthetic.json',file_reviewer=alice);candidate=store.resolve('1')
            context=packet(candidate,'support','The cited primary comparative remission result supports the trial-specific claim.',clinician=alice)
            context['sources']=[record()['source']]
            analysis={'summary':'Source result scoped to prediabetes remission.','reviewer_excerpt':context['review']['reason'],
                'proposed_decision':'support','assessments':[],'limitations':[],'source_assessments':[record()['facts']]}
            envelope={'context':context,'analysis':analysis};validate(analysis,context)
            saved=store.save_analysis('source-fixture',envelope)
            initial=store.confirm_source_analysis(candidate,alice,saved['id'])['after']
            store.review(candidate['id'],candidate['current_version'],alice,'support','My research judgment')
            store.review(candidate['id'],candidate['current_version'],bob,'oppose','I disagree without changing study facts')
            self.assertEqual(initial['score_100'],store.latest_evidence_score(candidate)['score_100'])
            self.assertEqual(initial['confidence'],store.latest_evidence_score(candidate)['confidence'])
            again=store.confirm_source_analysis(candidate,bob,saved['id']);self.assertFalse(again['changed'])
            self.assertEqual(len(store.evidence_events(candidate)),1)
            self.assertEqual(len(store.current_reviews(candidate)),2)
            head=store.evidence_digest(store.source_records(candidate))
            with patch('app.services.feedback_evidence.score_sources',side_effect=NativePLNError('Native failed')):
                with self.assertRaises(NativePLNError):store.correct_source(candidate,bob,'PMID:42655937','effect.direction','lower','Synthetic wrong-direction regression check',QUOTE)
            self.assertEqual(len(store.evidence_events(candidate)),1)
            changed=store.correct_source(candidate,bob,'PMID:42655937','effect.direction','lower','Deliberately wrong direction for rollback test',QUOTE,expected_digest=head)
            self.assertLess(changed['after']['score_100'],initial['score_100'])
            with self.assertRaisesRegex(ValueError,'Another reviewer changed'):
                store.correct_source(candidate,alice,'PMID:42655937','effect.direction','higher','Stale form',QUOTE,expected_digest=head)
            restored=store.restore_source_event(candidate,alice,initial['event_id'],'Correct direction is higher remission frequency')
            self.assertEqual(restored['after']['score_100'],initial['score_100'])
            self.assertEqual(len(store.evidence_events(candidate)),3)
            export=Path(directory)/'export.json';store.export(export);payload=json.loads(export.read_text())
            self.assertEqual(payload['original_handoff'],data)
            self.assertEqual(len(payload['records'][0]['source_evidence_events']),3)
            self.assertEqual(len(payload['records'][0]['current_reviews']),2)
            store.close();store=FeedbackStore(path)
            self.assertEqual(store.latest_evidence_score(candidate)['score_100'],initial['score_100'])
            # A source change invalidates the score; old facts/history survive.
            data['risk_areas'][0]['investigational_evidence'][0]['why']='Changed source snapshot'
            store.import_data(data,'changed.json',file_reviewer=alice)
            self.assertIsNone(store.latest_evidence_score(store.resolve('1')))
            self.assertEqual(len(store.evidence_events(store.resolve('1'))),3)
            store.close()

if __name__=='__main__':unittest.main()