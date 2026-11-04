"""Reason-specific native PLN inputs, consent, cache and ranking boundaries."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from app.services.feedback_auth import register_local, login_local
from app.services.feedback_omega import packet
from app.services.feedback_reason_schema import validate, apply_reason_policy
from app.services.feedback_store import FeedbackStore


def data():
    return {'patient_id':'synthetic-person-id','risk_areas':[
        {'disease_name':'Demo Condition','investigational_evidence':[
            {'name':'Demo A','why':'Synthetic evidence only.','publications':[]},
            {'name':'Demo B','why':'Synthetic comparator evidence.','publications':[]},
            {'name':'Demo Unassessed','why':'Synthetic missing evidence.','publications':[]}]},
        {'disease_name':'Other Demo Condition','treatments':[{'name':'Demo A','why':'Different context.'}]}]}


def interpreted(candidate, clinician, submitted='support', reason='I oppose retention: insufficient human evidence.', proposed='oppose'):
    context=packet(candidate,submitted,reason,clinician=clinician)
    context['sources']=[{'id':'handoff:why','text':candidate['data']['why']}]
    analysis={'summary':'The reason opposes retention and challenges the available human evidence.',
              'proposed_decision':proposed,'reviewer_excerpt':reason,
              'assessments':[{'aspect':'human_applicability','decision':'oppose','basis':'applicability_mismatch','explanation':'Human support is challenged.',
                              'reviewer_excerpt':reason,'source_checks':[]}], 'limitations':['Synthetic text only.'],'source_assessments':[]}
    return {'analysis':analysis,'context':context,'runtime':{'model':'synthetic-only'},'usage':{'model_calls':0}}


class FeedbackReasonChecks(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.path=Path(self.directory.name)/'reviews.sqlite3'
        self.store=FeedbackStore(self.path)
        for name in ('alice','bob'):
            register_local(self.store,name,name+'@example.test','Synthetic-pass-2026')
        self.alice=login_local(self.store,'alice@example.test','Synthetic-pass-2026').clinician
        self.bob=login_local(self.store,'bob@example.test','Synthetic-pass-2026').clinician
        self.store.import_data(data(),'synthetic.json',file_reviewer=self.alice)

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def test_confirmation_records_interpreted_opinion_without_fabricating_source_evidence(self):
        c=self.store.resolve('1')
        reason='I oppose retention: insufficient human evidence.'
        envelope=interpreted(c,self.alice,reason=reason)
        saved=self.store.save_analysis('synthetic-cache-key',envelope)
        result=self.store.review(c['id'],c['current_version'],self.alice,'oppose',reason,
            analysis_id=saved['id'],analysis_confirmed=True,submitted_decision='support')['after']
        self.assertIsNone(result)
        self.assertEqual(self.store.current_reviews(c)[0]['decision'],'oppose')
        self.assertEqual(self.store.current_reviews(c)[0]['submitted_decision'],'support')

    def test_declining_interpretation_preserves_original_judgment_and_adds_no_aspect(self):
        c=self.store.resolve('1')
        reason='I oppose retention: insufficient human evidence.'
        saved=self.store.save_analysis('synthetic-cache',interpreted(c,self.alice,reason=reason))
        result=self.store.review(c['id'],c['current_version'],self.alice,'support',reason,
            analysis_id=saved['id'],analysis_confirmed=False,submitted_decision='support')['after']
        self.assertIsNone(result)
        self.assertFalse(self.store.current_reviews(c)[0]['analysis_confirmed'])

    def test_clinician_can_explicitly_correct_the_proposed_final_judgment(self):
        c=self.store.resolve('1');reason='I oppose retention: insufficient human evidence.'
        envelope=interpreted(c,self.alice,reason=reason,proposed='support')
        saved=self.store.save_analysis('explicit-override',envelope)
        with self.assertRaisesRegex(ValueError,'Confirm the proposed'):
            self.store.review(c['id'],c['current_version'],self.alice,'oppose',reason,
                analysis_id=saved['id'],analysis_confirmed=True,submitted_decision='support')
        result=self.store.review(c['id'],c['current_version'],self.alice,'oppose',reason,
                analysis_id=saved['id'],analysis_confirmed=True,submitted_decision='support',decision_override=True)
        self.assertIsNone(result['after'])
        self.assertEqual(self.store.current_reviews(c)[0]['decision'],'oppose')
        self.assertTrue(self.store.current_reviews(c)[0]['decision_override'])

    def test_interpretation_cannot_be_replayed_on_another_claim_or_reason(self):
        c=self.store.resolve('1')
        saved=self.store.save_analysis('synthetic-cache',interpreted(c,self.alice))
        for other,reason in ((self.store.resolve('2'),'I oppose retention: insufficient human evidence.'),(c,'A different reason')):
            with self.assertRaisesRegex(ValueError,'another review'):
                self.store.review(other['id'],other['current_version'],self.alice,'oppose',reason,
                                  analysis_id=saved['id'],analysis_confirmed=True,submitted_decision='support')
        self.assertEqual(self.store.history(c)['reviews'],[])

    def test_draft_and_cache_survive_restart_and_repeat_does_not_duplicate(self):
        c=self.store.resolve('1')
        reason='I oppose retention: insufficient human evidence.'
        self.store.save_draft(c,self.alice,'support',reason,'')
        saved=self.store.save_analysis('synthetic-key',interpreted(c,self.alice,reason=reason))
        self.store.close();self.store=FeedbackStore(self.path)
        self.assertEqual(self.store.review_draft(c,self.alice)['rationale'],reason)
        self.assertEqual(self.store.cached_analysis('synthetic-key')['id'],saved['id'])
        for expected in (True,False):
            self.store.save_draft(c,self.alice,'support',reason,'')
            result=self.store.review(c['id'],c['current_version'],self.alice,'oppose',reason,
                analysis_id=saved['id'],analysis_confirmed=True,submitted_decision='support')
            self.assertEqual(result['changed'],expected)
            self.assertIsNone(self.store.review_draft(c,self.alice))
        self.assertEqual(len(self.store.history(c)['reviews']),1)

    def test_consent_is_per_clinician_dataset_and_destination(self):
        identity={'model':'demo-model','provider':'demo-provider','destination':'example.test'}
        self.assertFalse(self.store.has_external_consent(self.alice,identity))
        self.store.record_external_consent(self.alice,identity)
        self.assertTrue(self.store.has_external_consent(self.alice,identity))
        self.assertFalse(self.store.has_external_consent(self.bob,identity))
        self.assertFalse(self.store.has_external_consent(self.alice,{**identity,'destination':'different.example.test'}))
        new=data();new['patient_id']='another-synthetic-person'
        self.store.import_data(new,'new.json',file_reviewer=self.alice)
        self.assertFalse(self.store.has_external_consent(self.alice,identity))

    def test_rankings_change_after_revision_keep_conditions_separate_and_unknown_unranked(self):
        def review(index,who,decision):
            c=self.store.resolve(str(index));return self.store.review(c['id'],c['current_version'],who,decision,'Synthetic judgment','PMID:12345')
        review(1,self.alice,'support');review(2,self.alice,'support')
        review(1,self.bob,'oppose');review(4,self.alice,'oppose')
        groups=self.store.rankings()
        self.assertEqual([x['candidate']['position'] for x in groups[0]['ranked']],[2,1])
        self.assertEqual(len(groups[0]['unranked']),1)
        self.assertEqual(len(groups[1]['ranked']),1)
        self.assertEqual(groups[0]['ranked'][1]['assessment']['strength'],0.5)
        review(1,self.alice,'withdrawn')
        self.assertIsNone(self.store.latest_evidence_score(self.store.resolve('1')))

    def test_missing_effect_stays_unknown_instead_of_becoming_negative_effect(self):
        c=self.store.resolve('1')
        reason='There is not enough evidence; the biological effect is unknown.'
        envelope=interpreted(c,self.alice,submitted='oppose',reason=reason)
        envelope['analysis']['assessments']=[
            {'aspect':'evidence_sufficiency','decision':'oppose','basis':'evidence_gap','explanation':'Insufficient support.','reviewer_excerpt':reason,'source_checks':[]},
            {'aspect':'benefit_claim','decision':'uncertain','basis':'evidence_gap','explanation':'Effect unknown.','reviewer_excerpt':reason,'source_checks':[]}]
        saved=self.store.save_analysis('missing-effect',envelope)
        result=self.store.review(c['id'],c['current_version'],self.alice,'oppose',reason,
            analysis_id=saved['id'],analysis_confirmed=True,submitted_decision='oppose')['after']
        self.assertIsNone(result)
        self.assertEqual(self.store.current_reviews(c)[0]['reason_analysis']['analysis']['assessments'][1]['decision'],'uncertain')

    def test_source_quote_validation_and_numeric_weights_are_rejected(self):
        c=self.store.resolve('1');result=interpreted(c,self.alice)
        result['analysis']['assessments'][0]['source_checks']=[{'source_id':'handoff:why','quote':'Not in source'}]
        with self.assertRaisesRegex(ValueError,'not in the inspected'):
            validate(result['analysis'],result['context'])
        result=interpreted(c,self.alice);result['analysis']['weight']=0.9
        with self.assertRaisesRegex(ValueError,'unexpected'):
            validate(result['analysis'],result['context'])

    def test_typed_evidence_gap_rule_keeps_unknown_effect_out_of_negative_pln_premises(self):
        c=self.store.resolve('1');result=interpreted(c,self.alice)
        row=result['analysis']['assessments'][0]
        row.update(aspect='benefit_claim',basis='evidence_gap',decision='oppose')
        with self.assertRaisesRegex(ValueError,'Missing evidence'):
            validate(result['analysis'],result['context'])
        normalized,trace=apply_reason_policy(result['analysis'])
        self.assertEqual(normalized['assessments'][0]['decision'],'uncertain')
        self.assertEqual(trace[0]['from'],'oppose')
        self.assertEqual(result['analysis']['assessments'][0]['decision'],'oppose')
        validate(normalized,result['context'])

    def test_outbound_packet_omits_patient_accounts_and_redacts_known_identifiers(self):
        c=self.store.resolve('1')
        reason='Record '+c['patient_id']+' belongs to '+self.alice.email+'.'
        outbound=packet(c,'uncertain',reason,clinician=self.alice)
        wire=json.dumps(outbound)
        for private in (c['patient_id'],self.alice.email,self.alice.id,self.alice.subject):
            self.assertNotIn(private,wire)
        self.assertNotIn('patient_id',outbound)


if __name__=='__main__':unittest.main()
