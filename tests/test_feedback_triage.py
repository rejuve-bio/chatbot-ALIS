"""Unknown judgments, review coverage, attributed corrections and ranking."""
from pathlib import Path
import tempfile
import unittest

from app.services.feedback_beliefs import assess
from app.services.feedback_triage import review_status
from app.services.feedback_store import FeedbackStore
from app.services.feedback_auth import register_local,login_local


def observations(s,o,u):
    return [dict(id=i,reviewer_key='reviewer-'+str(i),reviewer='Reviewer '+str(i),decision=value,
                 rationale='',citation='',reason_mode='note')
            for i,value in enumerate(['support']*s+['oppose']*o+['uncertain']*u,1)]


class ReviewRouting(unittest.TestCase):
    def test_uncertain_is_not_a_half_vote_or_negative_evidence(self):
        first=assess(observations(2,0,0));with_unknown=assess(observations(2,0,1))
        self.assertEqual((first['strength'],first['confidence']),(with_unknown['strength'],with_unknown['confidence']))
        t=review_status(with_unknown,observations(2,0,1))
        self.assertEqual((t['decisive'],t['total'],t['uncertain']),(2,3,1))
        unknown=assess(observations(0,0,1000))
        self.assertIsNone(unknown['strength']);self.assertIsNone(unknown['confidence'])
        self.assertEqual(review_status(unknown,observations(0,0,1000))['status'],'needs_review')

    def test_many_uncertain_reviews_prevent_a_misleading_supported_status(self):
        reviews=observations(3,0,997);value=assess(reviews);t=review_status(value,reviews)
        self.assertEqual(value['strength'],1)
        self.assertEqual(value['confidence'],.75)
        self.assertEqual(t['status'],'needs_review')
        self.assertEqual(t['decisive_fraction'],.003)

    def test_support_opposition_disagreement_and_coverage_route_transparently(self):
        for counts,status in [((700,300,0),'more_support'),((300,700,0),'lower_support'),((500,500,0),'needs_review'),((0,0,0),'unreviewed')]:
            rows=observations(*counts)
            self.assertEqual(review_status({},rows)['status'],status)
        rows=observations(3,0,0)
        self.assertEqual(review_status({},rows,min_decisive=100)['status'],'needs_review')
        self.assertEqual(review_status({},rows,min_decisive=0)['status'],'more_support')

    def test_concern_flags_do_not_change_pln_but_remain_attributed(self):
        rows=observations(10,0,0);base=assess(rows)
        rows[0].update(concern_kind='patient_context',concern_text='Check applicability to this patient.')
        flagged=assess(rows);t=review_status(flagged,rows)
        self.assertEqual((base['strength'],base['confidence']),(flagged['strength'],flagged['confidence']))
        self.assertEqual(t['status'],'needs_review')
        self.assertEqual(t['concerns'][0]['reviewer'],'Reviewer 1')


class ReviewCorrections(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'reviews.sqlite3';self.store=FeedbackStore(self.path)
        self.people=[]
        for name in ('alice','bob','carol'):
            register_local(self.store,name,name+'@example.test','Synthetic-password-2026')
            self.people.append(login_local(self.store,name+'@example.test','Synthetic-password-2026').clinician)
        self.store.import_data({'patient_id':'synthetic','risk_areas':[{'disease_name':'Demo','investigational_evidence':[{'name':'A'},{'name':'B'}]}]},'demo.json',file_reviewer=self.people[0])

    def tearDown(self):self.store.close();self.tmp.cleanup()

    def submit(self,who,decision,target='1',**kw):
        c=self.store.resolve(target)
        return self.store.review(c['id'],c['current_version'],self.people[who],decision,'','',notes_only=True,**kw)

    def test_changing_oppose_to_uncertain_removes_its_decisive_weight(self):
        self.submit(0,'support');self.submit(1,'oppose')
        before=self.store.belief_assessment(self.store.resolve('1'))
        after=self.submit(1,'uncertain')['belief_after']
        self.assertEqual(before['strength'],.5);self.assertAlmostEqual(before['confidence'],2/3)
        self.assertEqual((after['strength'],after['confidence']),(1,.5))
        self.assertEqual(len(self.store.current_reviews(self.store.resolve('1'))),2)
        self.assertEqual(len(self.store.history(self.store.resolve('1'))['reviews']),3)

    def test_concern_edit_and_clear_are_versioned_without_changing_the_paper_or_count(self):
        self.submit(0,'support');c=self.store.resolve('1');original=c['data']
        flagged=self.submit(0,'support',concern_kind='evidence_mismatch',concern_text='This source concerns a different condition.')['belief_after']
        self.assertEqual((flagged['strength'],flagged['confidence']),(1,.5))
        self.submit(0,'uncertain')
        row=self.store.current_reviews(c)[0]
        self.assertEqual(row['concern_kind'],'evidence_mismatch')
        self.submit(0,'support',concern_kind='',concern_text='')
        self.assertEqual(self.store.current_reviews(c)[0]['concern_kind'],'')
        self.assertEqual(self.store.candidate(c['id'])['data'],original)
        self.assertTrue(any(r['concern_kind']=='evidence_mismatch' for r in self.store.history(c)['reviews']))
        self.store.close();self.store=FeedbackStore(self.path)
        self.assertEqual(self.store.current_reviews(self.store.resolve('1'))[0]['concern_kind'],'')

    def test_equal_support_prefers_more_review_coverage_without_a_product_score(self):
        self.submit(0,'support','1')
        for i in range(3):self.submit(i,'support','2')
        ranked=self.store.rankings()[0]['ranked']
        self.assertEqual([r['candidate']['position'] for r in ranked],[2,1])
        self.assertEqual([r['rank'] for r in ranked],[1,2])

    def test_withdrawn_concern_does_not_silently_reappear_on_a_new_review(self):
        self.submit(0,'uncertain',concern_kind='entry_error',concern_text='Check this entry.')
        self.submit(0,'withdrawn')
        self.submit(0,'support')
        self.assertEqual(self.store.current_reviews(self.store.resolve('1'))[0]['concern_kind'],'')

if __name__=='__main__':unittest.main()
