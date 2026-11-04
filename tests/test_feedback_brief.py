"""The optional Omega explanation cannot alter an appraisal or a clinician review."""
import copy
import json
from pathlib import Path
import runpy
import tempfile
import unittest

from app.services.feedback_auth import register_local,login_local,AuthenticationError
from app.services.feedback_brief import ReviewBrief,packet,reported_number_filter
from app.services.feedback_brief_schema import validate,schema_for,digest
from app.services.feedback_claims import source_digest
from app.services.feedback_store import FeedbackStore
from app.services.feedback_omega import OmegaReviewError

Session=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/feedback'))['Session']
SOURCE={'id':'PMID:12345','text':'Methods: Adults were studied. Results: Marker X increased in both groups. Clinical outcomes were not measured.',
        'abstract_available':True,'url':'https://pubmed.ncbi.nlm.nih.gov/12345/'}
SOURCE['text_sha256']=source_digest(SOURCE['text'])


def response(context):
    return dict(study_findings=[{'explanation':'The supplied study reports a marker change in both groups.',
                                'citations':[{'id':'PMID:12345','quote':'Marker X increased in both groups.'}]}],
                review_discussion=[],next_questions=[{'question':'Which outcome directly addresses the reviewed claim?',
                                                      'reference_ids':['PMID:12345']}])


class LocalOnlyClient:
    def __init__(self):self.calls=0
    def identity(self):return dict(model='synthetic-model',provider='test',destination='example.test')
    def brief_key(self,request):return digest(['test-runtime',request])
    def explain(self,request,progress=None):
        self.calls+=1
        return dict(brief=response(request),context=request,runtime={'model':'synthetic-model','core_commit':'test'},usage={'model_calls':1})


class BriefGuards(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.store=FeedbackStore(Path(self.tmp.name)/'reviews.sqlite3')
        for name in ('Alice','Bob'):register_local(self.store,name,name.lower()+'@example.test','Synthetic-password-2026')
        self.alice=login_local(self.store,'alice@example.test','Synthetic-password-2026')
        self.bob=login_local(self.store,'bob@example.test','Synthetic-password-2026')
        self.path=Path(self.tmp.name)/'handoff.json';self.path.write_text(json.dumps({'patient_id':'private-patient-id',
            'risk_areas':[{'disease_name':'Condition','investigational_evidence':[{'name':'Candidate',
                                                                                  'publications':[{'pmid':'12345'}]}]}]}))
        self.session=Session(self.store,self.alice,'local','http://127.0.0.1:5001');self.session.import_handoff(self.path)
        self.c=self.store.resolve('1');self.store._save_claim_publication(self.c,SOURCE)
        self.client=LocalOnlyClient();self.brief=ReviewBrief(self.session,self.client)
        self.store.record_external_consent(self.alice.clinician,self.client.identity())

    def tearDown(self):self.store.close();self.tmp.cleanup()

    def review(self,person,decision,note=''):
        return self.store.review(self.c['id'],self.c['current_version'],person.clinician,decision,note,notes_only=True)

    def test_schema_citations_do_not_share_the_source_and_note_enum(self):
        self.review(self.alice,'support','Marker finding merits review.')
        context=self.brief.context(self.c,[SOURCE]);schema=schema_for(context)
        study=schema['properties']['study_findings']['items']['properties']['citations']['items']['properties']['id']['enum']
        review=schema['properties']['review_discussion']['items']['properties']['citations']['items']['properties']['id']['enum']
        self.assertEqual(study,['PMID:12345']);self.assertTrue(review[0].startswith('Review:#'))

    def test_fabricated_quotes_references_and_score_fields_are_rejected(self):
        context=self.brief.context(self.c,[SOURCE]);valid=response(context);validate(valid,context)
        for change in ('quote','id','score'):
            value=copy.deepcopy(valid)
            if change=='quote':value['study_findings'][0]['citations'][0]['quote']='Invented finding'
            elif change=='id':value['study_findings'][0]['citations'][0]['id']='PMID:99999'
            else:value['strength']=1
            with self.assertRaises(ValueError):validate(value,context)

    def test_computed_numbers_are_omitted_from_explanations_without_editing_the_raw_output(self):
        context=self.brief.context(self.c,[SOURCE]);raw=response(context)
        raw['next_questions'][0]['question']='Does a calculated 32% dropout rate limit interpretation?'
        filtered,omissions=reported_number_filter(raw,context)
        self.assertEqual(filtered['next_questions'],[])
        self.assertEqual(len(omissions),1)
        self.assertIn('32%',raw['next_questions'][0]['question'])

    def test_success_is_cached_without_altering_reviews_appraisals_or_sources(self):
        self.review(self.alice,'support','A note for reviewers.')
        before=(self.store.current_reviews(self.c),self.store.belief_assessment(self.c),self.store.source_records(self.c))
        context=self.brief.context(self.c,[SOURCE]);first=self.brief.generate(self.c,[SOURCE],context)
        second=self.brief.generate(self.c,[SOURCE],context)
        self.assertFalse(first['cached']);self.assertTrue(second['cached']);self.assertEqual(self.client.calls,1)
        self.assertEqual(before,(self.store.current_reviews(self.c),self.store.belief_assessment(self.c),self.store.source_records(self.c)))

    def test_note_changes_invalidate_brief_while_numerical_pair_is_preserved(self):
        self.review(self.alice,'support','Initial note.');first=self.brief.context(self.c,[SOURCE]);a=self.store.belief_assessment(self.c)
        self.brief.generate(self.c,[SOURCE],first)
        self.review(self.alice,'support','Revised note.');second=self.brief.context(self.c,[SOURCE]);b=self.store.belief_assessment(self.c)
        self.assertNotEqual(self.brief.key(self.c,first),self.brief.key(self.c,second))
        self.assertIsNone(self.brief.cached(self.c,second))
        self.assertEqual((a['strength'],a['confidence']),(b['strength'],b['confidence']))

    def test_failure_does_not_block_a_later_review_or_change_prior_appraisal(self):
        self.review(self.alice,'support');a=self.store.belief_assessment(self.c)
        class Failure(LocalOnlyClient):
            def explain(self,*args,**kw):raise OmegaReviewError('Synthetic provider outage')
        brief=ReviewBrief(self.session,Failure())
        with self.assertRaises(OmegaReviewError):brief.generate(self.c,[SOURCE],brief.context(self.c,[SOURCE]))
        self.assertEqual(self.store.belief_assessment(self.c),a)
        self.review(self.bob,'oppose');self.assertEqual(self.store.belief_assessment(self.c)['strength'],.5)

    def test_changed_notes_during_generation_are_not_published_as_current(self):
        parent=self
        class Concurrent(LocalOnlyClient):
            def explain(self,request,progress=None):
                parent.review(parent.alice,'uncertain','New question during request.')
                return super().explain(request,progress)
        brief=ReviewBrief(self.session,Concurrent());context=brief.context(self.c,[SOURCE])
        with self.assertRaisesRegex(ValueError,'changed during'):brief.generate(self.c,[SOURCE],context)
        self.assertIsNone(brief.cached(self.c,context))

    def test_identifiers_and_names_are_redacted_before_model_packet(self):
        self.review(self.alice,'support','Alice mentioned private-patient-id and alice@example.test.')
        context=self.brief.context(self.c,[SOURCE]);raw=json.dumps(context)
        for private in ('private-patient-id','Alice','alice@example.test',self.alice.clinician.id):self.assertNotIn(private,raw)
        self.assertIn('Review:#',raw)

    def test_unauthorized_viewer_and_missing_consent_make_no_request(self):
        self.session.authentication=self.bob
        with self.assertRaises(AuthenticationError):self.brief.context(self.c,[SOURCE])
        self.session.import_handoff(self.path)
        context=self.brief.context(self.c,[SOURCE])
        with self.assertRaisesRegex(ValueError,'authorization'):self.brief.generate(self.c,[SOURCE],context)
        self.assertEqual(self.client.calls,0)


if __name__=='__main__':unittest.main()
