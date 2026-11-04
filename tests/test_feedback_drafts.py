"""Bad-draft recovery and budget guards; real CLI/native runs are separate."""
import copy
import contextlib
import io
import json
from pathlib import Path
import runpy
import tempfile
import time
import unittest
from unittest.mock import patch

from app.services.feedback_auth import register_local,login_local
from app.services.feedback_omega import OmegaReviewClient,OmegaReviewError,packet
from app.services.feedback_reason_schema import SCHEMA,schema_for,validate_review_reason,digest
from app.services.feedback_store import FeedbackStore

Session=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/feedback'))['Session']
REASON='The cited animal lifespan study does not report cancer-specific outcomes.'

class UnavailableOmega:
    def __init__(self):self.calls=0
    def identity(self):return {'model':'fixture-only','provider':'fixture','destination':'example.invalid'}
    def cache_key(self,request):return digest(request)
    def analyze(self,request,progress=None):
        self.calls+=1
        raise OmegaReviewError('Synthetic rejected quote.',{'error':'Synthetic rejected quote.'})

class ReasonInputChecks(unittest.TestCase):
    def test_judgments_numbers_and_bare_references_do_not_become_reasons(self):
        for value in ('2','1','3','', '  ','.','support','OPPOSE','PMID:38284547','https://example.test/paper'):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'write why'):
                validate_review_reason(value)
        self.assertEqual(validate_review_reason('Wrong species.'),'Wrong species.')

    def test_outbound_guard_runs_before_container_or_provider_access(self):
        with self.assertRaises(ValueError):packet({'data':{}},'support','2')
        client=object.__new__(OmegaReviewClient)
        with patch.object(client,'identity',side_effect=AssertionError('No container access expected')):
            with self.assertRaises(ValueError):client.analyze({'review':{'reason':'2'}})

    def test_schema_anchors_reason_fields_without_changing_the_static_contract(self):
        before=copy.deepcopy(SCHEMA)
        generated=schema_for({'review':{'reason':REASON}})
        self.assertEqual(generated['properties']['reviewer_excerpt']['enum'],[REASON])
        self.assertEqual(generated['properties']['assessments']['items']['properties']['reviewer_excerpt']['enum'],[REASON])
        self.assertEqual(SCHEMA,before)

class DraftRecovery(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.store=FeedbackStore(Path(self.tmp.name)/'reviews.sqlite3')
        register_local(self.store,'Test reviewer','reviewer@example.test','Synthetic-password-only')
        self.login=login_local(self.store,'reviewer@example.test','Synthetic-password-only')
        self.person=self.login.clinician
        self.store.import_data({'patient_id':'synthetic','risk_areas':[{'disease_name':'Cancer','investigational_evidence':[{'name':'Captopril','why':'Synthetic handoff for input validation.','publications':[]}]}]},'synthetic.json',file_reviewer=self.person)
        self.c=self.store.resolve('1')
        self.session=Session(self.store,self.login,'local','http://127.0.0.1:5001',omega=None)

    def tearDown(self):self.store.close();self.tmp.cleanup()

    def test_new_invalid_drafts_are_rejected_before_writing(self):
        with self.assertRaises(ValueError):self.store.save_draft(self.c,self.person,'support','2','')
        self.assertIsNone(self.store.review_draft(self.c,self.person))

    def test_legacy_support_dash_2_is_edited_before_review_submission(self):
        with self.store.transaction():
            self.store.db.execute('INSERT INTO review_drafts VALUES (?,?,?,?,?,?,?)',(self.c['id'],self.c['current_version'],self.person.id,'support','2','',time.time()))
        output=io.StringIO()
        with patch('builtins.input',side_effect=['2',REASON,'PMID:38284547']),contextlib.redirect_stdout(output):
            self.session.review(self.c,[])
        saved=self.store.current_reviews(self.c)
        self.assertEqual(saved[0]['decision'],'oppose');self.assertEqual(saved[0]['rationale'],REASON)
        self.assertIn('Edit it below; no model call has been made',' '.join(output.getvalue().split()))
        self.assertIsNone(self.store.review_draft(self.c,self.person))

    def test_note_never_changes_selected_judgment(self):
        with patch('builtins.input',side_effect=['1','2','']),contextlib.redirect_stdout(io.StringIO()):
            self.session.review(self.c,[])
        row=self.store.current_reviews(self.c)[0]
        self.assertEqual((row['decision'],row['rationale'],row['reason_mode']),('support','2','note'))

    def test_optional_note_and_reference_can_be_omitted(self):
        with patch('builtins.input',side_effect=['3','','']),contextlib.redirect_stdout(io.StringIO()):
            self.session.review(self.c,[])
        row=self.store.current_reviews(self.c)[0]
        self.assertEqual((row['decision'],row['rationale'],row['citation']),('uncertain','',''))
        self.assertIsNone(self.store.belief_assessment(self.c)['strength'])

    def test_existing_draft_can_be_discarded_without_submitting_anything(self):
        self.store.save_draft(self.c,self.person,'oppose',REASON,'')
        with patch('builtins.input',return_value='3'),contextlib.redirect_stdout(io.StringIO()):self.session.review(self.c,[])
        self.assertIsNone(self.store.review_draft(self.c,self.person));self.assertEqual(self.store.current_reviews(self.c),[])

    def test_review_never_uses_omega_even_when_source_inspection_is_enabled(self):
        omega=UnavailableOmega();self.session.omega=omega
        with patch.object(self.session,'interpret_review',side_effect=AssertionError('No reason interpretation allowed')),contextlib.redirect_stdout(io.StringIO()):
            self.session.review(self.c,['support',REASON,'PMID:38284547'])
            self.session.review(self.c,['support','A completely different note. PMID:11111','PMID:38284547'])
        rows=self.store.current_reviews(self.c)
        self.assertEqual(omega.calls,0)
        self.assertEqual(len(rows),1)
        self.assertFalse(rows[0]['analysis_confirmed'])
        self.assertIsNone(rows[0]['analysis_id'])
        self.assertEqual(rows[0]['reason_mode'],'note')
        self.assertIsNone(self.store.latest_evidence_score(self.c))
        self.assertIsNone(self.store.review_draft(self.c,self.person))
        belief=self.store.belief_assessment(self.c)
        self.assertEqual((belief['strength'],belief['confidence'],belief['ground_count']),(1,.5,1))
        self.assertEqual(belief['groups'][0]['roots'],['PMID:38284547'])

    def test_legacy_review_retry_does_not_send_or_save_anything(self):
        self.session.omega=UnavailableOmega()
        with self.assertRaisesRegex(ValueError,'no longer call Omega'):
            self.session.review(self.c,['--retry'])
        self.assertEqual(self.store.current_reviews(self.c),[])
        self.assertEqual(self.session.omega.calls,0)

    def test_same_rejected_request_is_not_automatically_resubmitted(self):
        omega=UnavailableOmega();self.session.omega=omega
        self.store.record_external_consent(self.person,omega.identity())
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(OmegaReviewError):self.session.obtain_analysis(self.c,'oppose',REASON,'')
            with self.assertRaisesRegex(OmegaReviewError,'No new model call was made'):self.session.obtain_analysis(self.c,'oppose',REASON,'')
            self.assertEqual(omega.calls,1)
            with self.assertRaises(OmegaReviewError):self.session.obtain_analysis(self.c,'oppose',REASON,'',retry=True)
        self.assertEqual(omega.calls,2)
        self.assertEqual(self.store.current_reviews(self.c),[])

if __name__=='__main__':unittest.main()
