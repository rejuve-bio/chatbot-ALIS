"""Source-quoted wording approval, version isolation and exact-match memory reuse."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.services.feedback_auth import register_local,login_local
from app.services.feedback_claims import source_digest
from app.services.feedback_store import FeedbackStore

CLAIM='Candidate A warrants research on marker X in the population studied.'
QUOTE='The intervention group had a higher marker X level than the control group.'
TEXT='Title: Synthetic study\nAbstract: Adults were randomized. Results: '+QUOTE+' Clinical outcomes were not reported.'
SOURCE={'id':'PMID:12345','title':'Synthetic study','text':TEXT,'abstract_available':True,
        'url':'https://pubmed.ncbi.nlm.nih.gov/12345/','retrieval':'synthetic test fixture','text_sha256':source_digest(TEXT)}

class ClaimLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'review.sqlite3'
        self.store=FeedbackStore(self.path)
        for name in ('alice','bob'):
            register_local(self.store,name,name+'@example.test','Synthetic-password-2026')
        self.alice=login_local(self.store,'alice@example.test','Synthetic-password-2026').clinician
        self.bob=login_local(self.store,'bob@example.test','Synthetic-password-2026').clinician
        self.data={'patient_id':'synthetic-A','risk_areas':[{'disease_name':'Condition A',
                   'investigational_evidence':[{'name':'Candidate A','drugs':['Compound A'],
                     'why':'A broad upstream rationale.','publications':[{'pmid':'12345'}]}]}]}
        self.store.import_data(self.data,'synthetic.json',file_reviewer=self.alice)
        self.c=self.store.resolve('1')
        self.store._save_claim_publication(self.c,SOURCE)

    def tearDown(self):self.store.close();self.tmp.cleanup()

    def propose(self):
        return self.store.propose_claim(self.c,self.alice,CLAIM,'PMID:12345',QUOTE)

    def approve(self):
        return self.store.resolve_correction(self.propose(),self.bob,True,'Checked wording against the synthetic source.')

    def test_proposal_changes_nothing_until_another_clinician_approves(self):
        self.store.review(self.c['id'],self.c['current_version'],self.alice,'support','Demo','',notes_only=True)
        before=self.c['current_version'];old_proposition=self.c['proposition']
        correction=self.propose()
        self.assertEqual(self.store.resolve('1')['proposition'],old_proposition)
        with self.assertRaisesRegex(ValueError,'different authenticated'):
            self.store.resolve_correction(correction,self.alice,True,'Self approval')
        result=self.store.resolve_correction(correction,self.bob,True,'Checked source scope')
        current=self.store.resolve('1')
        self.assertNotEqual(current['current_version'],before)
        self.assertEqual(current['proposition'],CLAIM)
        self.assertTrue(current['claim_approved'])
        self.assertEqual(current['data']['review_claim_grounding']['quote'],QUOTE)
        self.assertEqual(self.store.current_reviews(current),[])
        self.assertIsNone(self.store.belief_assessment(current)['strength'])
        self.assertEqual(len(self.store.history(current)['reviews']),1)
        self.assertEqual(result['knowledge_id'],1)

    def test_same_evidence_later_handoff_reuses_claim_but_not_patient_reviews(self):
        self.approve();first=self.store.resolve('1')
        self.store.review(first['id'],first['current_version'],self.bob,'support','Demo','',notes_only=True)
        other=copy.deepcopy(self.data);other['patient_id']='synthetic-B'
        with patch('app.services.feedback_claims.fetch_abstract',side_effect=AssertionError('No retrieval needed')):
            result=self.store.import_data(other,'new.json',file_reviewer=self.bob)
        current=self.store.resolve('1')
        self.assertEqual(result['learned'],1)
        self.assertEqual(current['proposition'],CLAIM)
        self.assertEqual(current['data']['review_claim_grounding']['source_text_sha256'],SOURCE['text_sha256'])
        self.assertEqual(self.store.current_reviews(current),[])
        self.assertEqual(self.store.learning_details(current)[0]['approver'],'bob')
        self.assertEqual(self.store._source_data(current),self.data['risk_areas'][0]['investigational_evidence'][0])
        self.store.close();self.store=FeedbackStore(self.path)
        self.assertEqual(self.store.resolve('1')['proposition'],CLAIM)

    def test_changed_source_or_condition_does_not_reuse_the_claim(self):
        self.approve()
        for change in ('source','condition','compound'):
            data=copy.deepcopy(self.data);data['patient_id']='synthetic-'+change
            if change=='source':data['risk_areas'][0]['investigational_evidence'][0]['publications'][0]['pmid']='67890'
            elif change=='condition':data['risk_areas'][0]['disease_name']='Condition B'
            else:data['risk_areas'][0]['investigational_evidence'][0]['drugs']=['Compound B']
            result=self.store.import_data(data,change+'.json',file_reviewer=self.alice)
            self.assertEqual(result['learned'],0)
            self.assertFalse(self.store.resolve('1')['claim_approved'])

    def test_invented_quote_unlinked_paper_or_missing_grounding_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'quote must match'):
            self.store.propose_claim(self.c,self.alice,CLAIM,'PMID:12345','Invented marker improvement with no source support.')
        with self.assertRaisesRegex(ValueError,'supplied with'):
            self.store.propose_claim(self.c,self.alice,CLAIM,'PMID:67890',QUOTE)
        with self.assertRaisesRegex(ValueError,'Use claim N'):
            self.store.propose_correction(self.c['id'],self.c['current_version'],self.alice,'review_claim',CLAIM,'Reason','PMID:12345',reusable=True)
        self.assertEqual(self.store.claim_proposals(self.c),[])

    def test_rejection_and_stale_approval_leave_current_claim_intact(self):
        correction=self.propose()
        self.store.resolve_correction(correction,self.bob,False,'Scope not acceptable')
        self.assertFalse(self.store.resolve('1')['claim_approved'])
        stale=self.propose()
        another=self.store.propose_correction(self.c['id'],self.c['current_version'],self.alice,'why','Changed rationale','Reason')
        self.store.resolve_correction(another,self.bob,True,'Checked')
        with self.assertRaisesRegex(ValueError,'older evidence'):
            self.store.resolve_correction(stale,self.bob,True,'Late approval')

    def test_retirement_restores_source_statement_and_keeps_claim_history(self):
        self.approve()
        current=self.store.resolve('1')
        self.assertEqual(self.store.retire_knowledge(1,self.bob,'Withdraw shared wording'),1)
        current=self.store.resolve('1')
        self.assertFalse(current['claim_approved'])
        self.assertNotIn('review_claim',current['data'])
        self.assertTrue(any(v['statement']==CLAIM for v in self.store.claim_history(current)))

    def test_saved_source_integrity_is_checked_before_approval(self):
        identity=self.propose()
        altered=copy.deepcopy(SOURCE);altered['text']='Modified public-source cache'
        self.store.db.execute('UPDATE claim_publications SET source_json=?',(json.dumps(altered),))
        with self.assertRaisesRegex(ValueError,'integrity'):
            self.store.resolve_correction(identity,self.bob,True,'Attempt approval')
        self.assertEqual(self.store.claim_proposals(self.c)[0]['status'],'pending')

if __name__=='__main__':unittest.main()
