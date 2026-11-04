"""Native appraisal revision, provenance grouping and durable lifecycle checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.services.feedback_beliefs import assess, assess_reference_groups, reference_ids
from app.services.feedback_pln import NativePLNError
from app.services.feedback_store import FeedbackStore
from app.services.feedback_auth import register_local, login_local


def review(n, decision, citation="PMID:12345"):
    return dict(id=n, reviewer_key="account-"+str(n), decision=decision,
                rationale="Synthetic research appraisal.", citation=citation,
                analysis_confirmed=False, reason_analysis=None)


class ArchivedReferencePolicyChecks(unittest.TestCase):
    def test_same_reference_accounts_do_not_inflate_confidence(self):
        first=assess_reference_groups([review(1,"support")])
        repeated=assess_reference_groups([review(n,"support") for n in range(1,6)])
        self.assertEqual((first['strength'],first['confidence']),(1,.5))
        self.assertEqual((repeated['strength'],repeated['confidence']),(1,.5))
        contested=assess_reference_groups([review(1,'support'),review(2,'oppose')])
        self.assertEqual((contested['strength'],contested['confidence']),(.5,.5))
        self.assertTrue(contested['conflict'])

    def test_distinct_grounds_revise_and_a_bridge_removes_double_count(self):
        independent=assess_reference_groups([review(1,'support','PMID:12345'),review(2,'oppose','PMID:67890')])
        self.assertEqual(independent['strength'],.5)
        self.assertAlmostEqual(independent['confidence'],2/3)
        linked=assess_reference_groups([review(1,'support','PMID:12345'),review(2,'oppose','PMID:67890')],
                      {'PMID:12345':['STUDY:NCT00000001'],'PMID:67890':['STUDY:NCT00000001']})
        self.assertEqual(linked['confidence'],.5)
        transitive=assess_reference_groups([review(1,'support','PMID:12345 PMID:22222'),
                           review(2,'oppose','PMID:22222 PMID:67890'),review(3,'support','PMID:67890')])
        self.assertEqual(transitive['ground_count'],1)

    def test_unknown_is_not_default_half_strength(self):
        with patch('app.services.feedback_beliefs.execute_native',side_effect=AssertionError('Unexpected numerical call')):
            self.assertIsNone(assess_reference_groups([])['strength'])
            value=assess_reference_groups([review(1,'uncertain')])
            self.assertIsNone(value['strength']);self.assertIsNone(value['confidence'])

    def test_unreferenced_decision_is_saved_without_inventing_a_ground(self):
        with patch('app.services.feedback_beliefs.execute_native',side_effect=AssertionError('Unexpected numerical call')):
            value=assess_reference_groups([review(1,'support','')])
        self.assertEqual(value['status'],'needs_reference')
        self.assertEqual(value['support'],1)
        self.assertEqual(value['ground_count'],0)
        self.assertIsNone(value['strength'])

    def test_confirmed_reason_updates_only_its_scoped_claims(self):
        item=review(1,'oppose')
        item['analysis_confirmed']=True
        item['reason_analysis']={'analysis':{'assessments':[
            {'aspect':'evidence_sufficiency','decision':'oppose','basis':'evidence_gap','source_checks':[]},
            {'aspect':'benefit_claim','decision':'oppose','basis':'evidence_gap','source_checks':[]}]}}
        value=assess_reference_groups([item])
        self.assertEqual(value['strength'],0)
        self.assertEqual(value['aspect_assessments']['evidence_sufficiency']['strength'],0)
        self.assertIsNone(value['aspect_assessments']['benefit_claim']['strength'])
        self.assertNotIn('patient_suitability',value['aspect_assessments'])
        item['analysis_confirmed']=False
        self.assertEqual(assess_reference_groups([item])['aspect_assessments'],{})

    def test_shared_note_text_and_identifiers_do_not_affect_appraisal(self):
        item=review(1,'support')
        item['reason_mode']='note'
        first=assess_reference_groups([item])
        item['rationale']='Strongly oppose. PMID:67890 DOI:10.1000/example'
        # Even an accidentally attached old analysis cannot make a note an aspect premise.
        item.update(analysis_confirmed=True,reason_analysis={'analysis':{'assessments':[
            {'aspect':'evidence_sufficiency','decision':'oppose','basis':'reported_effect',
             'source_checks':[{'source_id':'PMID:99999'}]}]}})
        changed=assess_reference_groups([item])
        for key in ('strength','confidence','ground_count','aspect_assessments'):
            self.assertEqual(first[key],changed[key])
        self.assertEqual(changed['groups'][0]['roots'],['PMID:12345'])
        item['citation']=''
        self.assertEqual(assess_reference_groups([item])['status'],'needs_reference')

    def test_reference_alias_and_duplicate_identity(self):
        self.assertEqual(reference_ids('https://pubmed.ncbi.nlm.nih.gov/12345/'),['PMID:12345'])
        self.assertEqual(reference_ids('DOI:10.1000/Example'),reference_ids('https://doi.org/10.1000/example'))
        checked=review(2,'support','')
        checked.update(analysis_confirmed=True,reason_analysis={'analysis':{'assessments':[
            {'aspect':'evidence_sufficiency','decision':'support','basis':'reported_effect',
             'source_checks':[{'source_id':'DOI:10.1000/EXAMPLE'}]}]}})
        value=assess_reference_groups([review(1,'support','https://doi.org/10.1000/example'),checked])
        self.assertEqual(value['ground_count'],1)
        with self.assertRaisesRegex(ValueError,'one current'):
            assess_reference_groups([review(1,'support'),review(1,'support')])

    def test_public_identifier_in_actual_reason_is_a_reference(self):
        item=review(1,'oppose','')
        item['rationale']='This study concerns lifespan, not the stated condition."\n"PMID:38284547'
        value=assess_reference_groups([item])
        self.assertEqual(value['groups'][0]['roots'],['PMID:38284547'])
        self.assertEqual((value['strength'],value['confidence']),(0,.5))
        other=review(2,'uncertain','PMID:3828454')
        value=assess_reference_groups([item,other])
        self.assertEqual(value['ground_count'],1)
        self.assertEqual(value['strength'],0)
        self.assertEqual(value['groups'][0]['roots'],['PMID:3828454'])

    def test_model_summary_alone_does_not_create_or_repair_a_reference(self):
        item=review(1,'oppose','')
        item['reason_analysis']={'analysis':{'summary':'Discussed PMID:38284547','assessments':[]}}
        item['analysis_confirmed']=True
        self.assertEqual(assess_reference_groups([item])['status'],'needs_reference')


class NativeReviewCountChecks(unittest.TestCase):
    def test_same_paper_three_reviewers_accumulate_count_confidence(self):
        for n in (1,2,3):
            value=assess([review(i,'support') for i in range(1,n+1)])
            self.assertEqual(value['strength'],1)
            self.assertAlmostEqual(value['confidence'],n/(n+1))
            self.assertEqual(value['decisive_count'],n)
            self.assertEqual(value['reference_group_count'],1)
            self.assertTrue(value['pln_used'])
            self.assertEqual(value['confidence_scope'],'decisive-review-count')
            if n>1:self.assertIn('Truth__Revision',value['native']['program'])

    def test_third_reviewer_opposition_is_preserved_with_larger_count(self):
        value=assess([review(1,'support'),review(2,'support'),review(3,'oppose')])
        self.assertAlmostEqual(value['strength'],2/3)
        self.assertEqual(value['confidence'],.75)
        self.assertTrue(value['conflict'])
        self.assertFalse(value['calibrated'])
        self.assertIsNone(value['scientific_confidence'])
        self.assertFalse(value['independence_assessed'])

    def test_uncertain_review_has_no_binary_weight(self):
        value=assess([review(1,'support'),review(2,'support'),review(3,'uncertain')])
        self.assertEqual(value['strength'],1)
        self.assertAlmostEqual(value['confidence'],2/3)
        self.assertEqual(value['reviewer_count'],3)
        self.assertEqual(value['decisive_count'],2)
        with patch('app.services.feedback_beliefs.execute_native',side_effect=AssertionError('No numeric premise')):
            self.assertIsNone(assess([review(1,'uncertain')])['strength'])
            self.assertIsNone(assess([])['confidence'])

    def test_reference_presence_and_study_overlap_do_not_change_review_weight(self):
        a=assess([review(1,'support',''),review(2,'oppose','')])
        b=assess([review(1,'support'),review(2,'oppose','PMID:67890')],
                 {'PMID:12345':['STUDY:NCT00000001'],'PMID:67890':['STUDY:NCT00000001']})
        self.assertEqual((a['strength'],a['confidence']),(b['strength'],b['confidence']))
        self.assertEqual(a['reference_group_count'],0)
        self.assertEqual(b['reference_group_count'],1)
        self.assertEqual(a['status'],'disagreement')

    def test_notes_and_previous_reason_interpretations_never_supply_new_premises(self):
        item=review(1,'support','')
        item.update(rationale='!(unsafe-note) PMID:99999',analysis_confirmed=True,
                    reason_analysis={'analysis':{'assessments':[
                        {'aspect':'benefit_claim','decision':'oppose','basis':'reported_effect',
                         'source_checks':[{'source_id':'PMID:88888'}]}]}})
        value=assess([item])
        self.assertEqual((value['strength'],value['confidence']),(1,.5))
        self.assertEqual(value['aspect_assessments'],{})
        self.assertEqual(value['reference_group_count'],0)
        self.assertNotIn('unsafe-note',value['native']['program'])
        self.assertNotIn('PMID:',value['native']['program'])

    def test_duplicate_account_or_invalid_current_decision_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'one current'):
            assess([review(1,'support'),review(1,'oppose')])
        with self.assertRaisesRegex(ValueError,'current Support'):
            assess([review(1,'withdrawn')])


class DurableAppraisalChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'reviews.sqlite3'
        self.store=FeedbackStore(self.path)
        for name in ('alice','bob'):
            register_local(self.store,name,name+'@example.test','Synthetic-password-2026')
        self.alice=login_local(self.store,'alice@example.test','Synthetic-password-2026').clinician
        self.bob=login_local(self.store,'bob@example.test','Synthetic-password-2026').clinician
        self.data={'patient_id':'synthetic-only','risk_areas':[{'disease_name':'Demo',
            'investigational_evidence':[{'name':'A','why':'Synthetic initial claim'},{'name':'B','why':'Another claim'}]}]}
        self.store.import_data(self.data,'synthetic.json',file_reviewer=self.alice)

    def tearDown(self):
        self.store.close();self.tmp.cleanup()

    def submit(self,clinician,decision,citation='PMID:12345'):
        c=self.store.resolve('1')
        return self.store.review(c['id'],c['current_version'],clinician,decision,'Synthetic reason for '+decision,citation)

    def test_replacement_withdrawal_restart_and_export(self):
        first=self.submit(self.alice,'support')['belief_after']
        self.submit(self.bob,'oppose')
        changed=self.submit(self.alice,'oppose')['belief_after']
        self.assertEqual(changed['strength'],0)
        self.assertEqual(changed['ground_count'],1)
        self.assertFalse(self.submit(self.alice,'oppose')['changed'])
        self.submit(self.bob,'withdrawn');self.submit(self.alice,'withdrawn')
        value=self.store.belief_assessment(self.store.resolve('1'))
        self.assertIsNone(value['strength'])
        self.assertEqual(len(self.store.history(self.store.resolve('1'))['reviews']),5)
        self.store.close();self.store=FeedbackStore(self.path)
        self.assertIsNone(self.store.belief_assessment(self.store.resolve('1'))['strength'])
        out=Path(self.tmp.name)/'export.json';self.store.export(out)
        payload=json.loads(out.read_text())
        self.assertEqual(payload['original_handoff'],self.data)
        self.assertIn('claim_appraisal',payload['records'][0])
        self.assertEqual(payload['records'][0]['belief_assessments'][1]['strength'],first['strength'])

    def test_native_failure_rolls_back_review_and_snapshot(self):
        c=self.store.resolve('1')
        before=len(self.store.belief_history(c))
        with patch('app.services.feedback_beliefs.execute_native',side_effect=NativePLNError('Unavailable')):
            with self.assertRaises(NativePLNError):self.submit(self.alice,'support')
        self.assertEqual(self.store.current_reviews(c),[])
        self.assertEqual(len(self.store.belief_history(c)),before)

    def test_changed_handoff_resets_beliefs_but_keeps_audit(self):
        self.submit(self.alice,'support')
        changed=copy.deepcopy(self.data)
        changed['risk_areas'][0]['investigational_evidence'][0]['why']='Changed evidence snapshot'
        self.store.import_data(changed,'new.json',file_reviewer=self.alice)
        c=self.store.resolve('1')
        self.assertIsNone(self.store.belief_assessment(c)['strength'])
        self.assertEqual(len(self.store.history(c)['reviews']),1)

    def test_version_seven_scores_are_rebuilt_without_changing_reviews(self):
        c=self.store.resolve('1')
        self.store.review(c['id'],c['current_version'],self.alice,'oppose',
                          'The cited research does not address this condition. PMID:38284547','')
        before=self.store.history(c)['reviews']
        # Reproduce an already saved v1 score that missed the embedded PMID.
        self.store.db.execute("UPDATE belief_assessments SET result_json=? WHERE candidate_id=?",
                              (json.dumps({'adapter':'reason-grounded-appraisal-v1','input_digest':'previous-policy',
                                           'strength':None,'confidence':None,'status':'needs_reference'}),c['id']))
        self.store.db.execute('PRAGMA user_version=7')
        self.store.close();self.store=FeedbackStore(self.path)
        current=self.store.belief_assessment(self.store.resolve('1'))
        self.assertEqual(current['strength'],0)
        self.assertEqual(current['confidence'],.5)
        self.assertEqual(before,self.store.history(c)['reviews'])
        self.assertEqual(self.store.db.execute('PRAGMA user_version').fetchone()[0],13)

    def test_existing_v9_reviews_rebuild_counts_and_keep_previous_policy_history(self):
        self.submit(self.alice,'support');self.submit(self.bob,'support')
        c=self.store.resolve('1')
        snapshot=self.store.history(c)['reviews']
        prior={'adapter':'reason-grounded-appraisal-v2','input_digest':'v9-policy',
               'strength':1,'confidence':.5,'status':'reviewed'}
        self.store.db.execute('UPDATE belief_assessments SET result_json=? WHERE id=(SELECT MAX(id) FROM belief_assessments WHERE candidate_id=?)',
                              (json.dumps(prior),c['id']))
        self.store.db.execute('PRAGMA user_version=9')
        self.store.close();self.store=FeedbackStore(self.path)
        c=self.store.resolve('1');current=self.store.belief_assessment(c)
        self.assertEqual(current['strength'],1)
        self.assertAlmostEqual(current['confidence'],2/3)
        self.assertEqual(snapshot,self.store.history(c)['reviews'])
        self.assertTrue(any(r['adapter']=='reason-grounded-appraisal-v2' for r in self.store.belief_history(c)))
        self.assertTrue(self.store.belief_policy_migrated)

    def test_ranking_does_not_reward_a_larger_confidence_product(self):
        self.submit(self.alice,'support')
        clinicians=[self.alice,self.bob]
        for name in ('carol','dave'):
            register_local(self.store,name,name+'@example.test','Synthetic-password-2026')
            clinicians.append(login_local(self.store,name+'@example.test','Synthetic-password-2026').clinician)
        c=self.store.resolve('2')
        for i,who in enumerate(clinicians):
            self.store.review(c['id'],c['current_version'],who,'support' if i<3 else 'oppose',
                              'Synthetic referenced appraisal.','PMID:'+str(11111+i))
        rows=self.store.rankings()[0]['ranked']
        self.assertEqual([r['candidate']['position'] for r in rows],[1,2])
        a,b=[r['assessment'] for r in rows]
        self.assertGreater(a['strength'],b['strength'])
        self.assertLess(a['strength']*a['confidence'],b['strength']*b['confidence'])


if __name__=='__main__':unittest.main()
