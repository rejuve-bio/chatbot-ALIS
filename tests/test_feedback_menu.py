"""The ordinary flow needs no commands, citations, model calls or proposal IDs."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

from app.services.feedback_auth import register_local,login_local,AuthenticationError
from app.services.feedback_claims import source_digest
from app.services.feedback_menu import GuidedFeedback,confidence_text
from app.services.feedback_store import FeedbackStore
from app.services.feedback_atlas import ReviewedAtlas
from tests.test_feedback_atlas import PolicyOnlyClient

Session=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/feedback'))['Session']
SOURCE_TEXT='Title: Synthetic study\nAbstract: Methods: Adults were randomized. Results: Marker X increased in the intervention group compared with the control group.'
SOURCE={'id':'PMID:12345','title':'Synthetic study','text':SOURCE_TEXT,'abstract_available':True,
        'url':'https://pubmed.ncbi.nlm.nih.gov/12345/','text_sha256':source_digest(SOURCE_TEXT)}

class GuidedMenuChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.store=FeedbackStore(Path(self.tmp.name)/'reviews.sqlite3')
        for name in ('alice','bob'):
            register_local(self.store,name,name+'@example.test','Synthetic-password-2026')
        self.alice=login_local(self.store,'alice@example.test','Synthetic-password-2026')
        self.bob=login_local(self.store,'bob@example.test','Synthetic-password-2026')
        self.path=Path(self.tmp.name)/'handoff.json'
        self.data={'patient_id':'synthetic','risk_areas':[{'disease_name':'Demo','investigational_evidence':[
            {'name':'Candidate A','publications':[{'pmid':'12345','title':'Synthetic study'}]}]}]}
        self.path.write_text(json.dumps(self.data))
        self.session=Session(self.store,self.alice,'local','http://127.0.0.1:5001')
        self.session.import_handoff(self.path)
        self.c=self.store.resolve('1');self.store._save_claim_publication(self.c,SOURCE)
        self.output=io.StringIO();self.menu=GuidedFeedback(self.session,lambda value='',indent=0:print(value,file=self.output),self.path)
        self.menu.atlas=ReviewedAtlas(self.session,PolicyOnlyClient())

    def tearDown(self):self.store.close();self.tmp.cleanup()

    def test_select_support_optional_note_saved_without_commands_or_model(self):
        with patch('builtins.input',side_effect=['1','1','Useful demo note.','q']),patch.object(self.session,'interpret_review',side_effect=AssertionError('No model call')):
            self.menu.run()
        row=self.store.current_reviews(self.c)[0]
        self.assertEqual((row['decision'],row['rationale'],row['citation']),('support','Useful demo note.','PMID:12345'))
        self.assertEqual(row['reason_mode'],'note')
        self.assertIn('After:',self.output.getvalue())
        self.assertNotIn('feedback>',self.output.getvalue())
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM feedback_analyses').fetchone()[0],0)

    def test_switch_reviewer_reopens_selected_file_with_the_new_identity(self):
        self.store.review(self.c['id'],self.c['current_version'],self.alice.clinician,'support','First reviewer',notes_only=True)
        with patch('app.services.feedback_menu.interactive_login',return_value=self.bob),patch('builtins.input',side_effect=['a','1','2','','q']):
            self.menu.run()
        current=self.store.current_reviews(self.store.resolve('1'))
        self.assertEqual({r['clinician_id'] for r in current},{self.alice.clinician.id,self.bob.clinician.id})
        self.assertIn('First reviewer',self.output.getvalue())
        self.assertAlmostEqual(self.store.belief_assessment(self.store.resolve('1'))['strength'],.5)
        self.assertAlmostEqual(self.store.belief_assessment(self.store.resolve('1'))['confidence'],2/3)

    def test_reported_correction_is_an_attributed_concern_not_a_paper_rewrite(self):
        with patch('builtins.input',side_effect=['1','5','1','This source may not address the stated condition.','3','0','q']):
            self.menu.run()
        current=self.store.current_reviews(self.store.resolve('1'))[0]
        self.assertEqual(current['decision'],'uncertain')
        self.assertEqual(current['concern_kind'],'evidence_mismatch')
        self.assertEqual(self.store.claim_proposals(self.c),[])
        self.assertNotIn('review_claim',self.store.resolve('1')['data'])
        self.assertIn('Needs review',self.output.getvalue())
        with patch('builtins.input',side_effect=['1','5','6','0','q']):self.menu.run()
        self.assertEqual(self.store.current_reviews(self.store.resolve('1'))[0]['concern_kind'],'')
        self.assertTrue(any(r['concern_kind'] for r in self.store.history(self.c)['reviews']))

    def test_filters_change_the_view_without_removing_records(self):
        with patch('builtins.input',side_effect=['v','2','q']):self.menu.run()
        self.assertIn('No entries match this view',self.output.getvalue())
        self.assertEqual(len(self.store.candidates()),1)
        self.assertEqual(self.menu.view,'more_support')
        with patch('builtins.input',side_effect=['v','6','100','q']):self.menu.run()
        self.assertEqual(self.menu.min_decisive,100)

    def test_approved_wording_uses_its_supporting_paper_not_first_in_bibliography(self):
        self.data['risk_areas'][0]['investigational_evidence'][0]['publications'].insert(0,{'pmid':'67890'})
        self.path.write_text(json.dumps(self.data));self.session.import_handoff(self.path)
        c=self.store.resolve('1')
        proposal=self.store.propose_claim(c,self.alice.clinician,'Candidate A warrants research on marker X.','PMID:12345','Marker X increased in the intervention group compared with the control group.')
        self.store.resolve_correction(proposal,self.bob.clinician,True,'Checked scope')
        with patch('app.services.feedback_claims.fetch_abstract',side_effect=AssertionError('Wrong source or unnecessary retrieval')):
            source=self.menu.load_source(self.store.resolve('1'))
        self.assertEqual(source['id'],'PMID:12345')

    def test_stale_statement_cannot_be_reviewed_after_another_person_changes_it(self):
        correction=self.store.propose_correction(self.c['id'],self.c['current_version'],self.alice.clinician,'why','New rationale','Reason')
        self.store.resolve_correction(correction,self.bob.clinician,True,'Checked')
        with patch('builtins.input',return_value=''),self.assertRaisesRegex(ValueError,'evidence changed|Evidence changed'):
            self.menu.submit(self.c,'support',SOURCE)
        self.assertEqual(self.store.current_reviews(self.store.resolve('1')),[])

    def test_unseen_file_requires_access_and_large_count_does_not_round_to_certainty(self):
        self.session.authentication=self.bob
        with self.assertRaises(AuthenticationError):self.menu.home()
        self.assertEqual(confidence_text(10000/10001),'0.999900')
        self.assertEqual(confidence_text(10000000/10000001),'<1.000000')

if __name__=='__main__':unittest.main()
