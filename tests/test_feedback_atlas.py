"""Persistence and context guards for actual review-driven atlas changes.

The native Core integration is exercised separately with the real container.
These tests isolate store/policy behavior and deliberately stub only dispatch.
"""
import json
from pathlib import Path
import runpy
import tempfile
import unittest

from app.services.feedback_auth import register_local, login_local, AuthenticationError
from app.services.feedback_atlas import ReviewedAtlas, AtlasError
from app.services.feedback_atlas_policy import assemble, digest
from app.services.feedback_store import FeedbackStore

Session = runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/feedback'))['Session']


class PolicyOnlyClient:
    def __init__(self): self.calls=0
    def run(self, context):
        self.calls+=1
        return dict(atlas=assemble(context), tool_trace=[], runtime={'execution':'test-policy-only'}, model_calls=0)


class AtlasLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.store=FeedbackStore(self.root/'reviews.sqlite3');self.people=[]
        for name in ('alice','bob','carol'):
            register_local(self.store,name,name+'@example.test','Synthetic-test-password')
            self.people.append(login_local(self.store,name+'@example.test','Synthetic-test-password'))
        self.data={'patient_id':'synthetic','risk_areas':[{'disease_name':'Condition',
            'investigational_evidence':[{'name':'First'},{'name':'Second'},{'name':'Third'}]}]}
        self.path=self.root/'handoff.json';self.path.write_text(json.dumps(self.data))
        self.session=Session(self.store,self.people[0],'local','http://127.0.0.1:5001')
        self.session.import_handoff(self.path);self.client=PolicyOnlyClient()
        self.atlas=ReviewedAtlas(self.session,self.client)

    def tearDown(self):self.store.close();self.tmp.cleanup()

    def review(self,index,decision,entry='2',note='',**kw):
        c=self.store.resolve(entry)
        return self.store.review(c['id'],c['current_version'],self.people[index].clinician,decision,note,notes_only=True,**kw)

    def row(self,result,number=2):return next(r for r in result['atlas']['entries'] if r['entry_number']==number)

    def test_atlas_changes_and_explains_cause_then_reuses_persisted_result(self):
        first=self.atlas.refresh();self.assertEqual(self.row(first)['research_order'],2)
        self.review(0,'support');second=self.atlas.refresh()
        self.assertEqual(self.row(second)['research_order'],1)
        change=next(c for c in second['changes'] if c['entry_number']==2)
        self.assertEqual(change['cause_events'][0]['reviewer'],'alice')
        self.assertEqual(change['before']['status'],'Not reviewed')
        self.assertEqual(change['after']['status'],'More support')
        repeated=self.atlas.refresh();self.assertFalse(repeated['rebuilt']);self.assertEqual(self.client.calls,2)
        self.store.close();self.store=FeedbackStore(self.root/'reviews.sqlite3')
        self.session.store=self.store
        for auth in self.people:auth.store=self.store
        self.assertEqual(ReviewedAtlas(self.session,self.client).refresh()['snapshot_id'],second['snapshot_id'])

    def test_opposition_uncertainty_and_concerns_change_atlas_without_adding_false_weight(self):
        self.atlas.refresh();self.review(0,'support');self.atlas.refresh()
        self.review(1,'uncertain');uncertain=self.atlas.refresh()
        self.assertEqual(self.row(uncertain)['routing']['status'],'needs_review')
        self.assertEqual(self.row(uncertain)['appraisal']['confidence'],.5)
        self.review(1,'oppose');self.review(2,'oppose');opposed=self.atlas.refresh()
        self.assertEqual(self.row(opposed)['research_order'],3)
        self.assertEqual(self.row(opposed)['routing']['status'],'lower_support')
        self.review(1,'support');supported=self.atlas.refresh()
        self.review(0,'support',concern_kind='safety_context',concern_text='Check supplied interaction information.')
        flagged=self.atlas.refresh()
        self.assertEqual(self.row(flagged)['appraisal']['strength'],self.row(supported)['appraisal']['strength'])
        self.assertEqual(self.row(flagged)['routing']['status'],'needs_review')
        self.assertEqual(self.row(flagged)['routing']['concerns'][0]['reviewer'],'alice')

    def test_note_only_change_is_audited_and_duplicate_does_not_create_snapshot(self):
        self.review(0,'support');self.atlas.refresh()
        self.review(0,'support',note='Shared context only.');result=self.atlas.refresh()
        change=next(c for c in result['changes'] if c['entry_number']==2)
        self.assertEqual(change['before'],change['after'])
        self.assertEqual(change['cause_events'][0]['note'],'Shared context only.')
        self.review(0,'support',note='Shared context only.')
        self.assertEqual(self.atlas.refresh()['snapshot_id'],result['snapshot_id'])

    def test_new_patient_and_evidence_version_do_not_inherit_reviews(self):
        self.review(0,'support');self.atlas.refresh()
        self.data['risk_areas'][0]['investigational_evidence'][1]['why']='Changed source interpretation'
        self.path.write_text(json.dumps(self.data));self.session.import_handoff(self.path)
        changed=self.atlas.refresh();self.assertEqual(self.row(changed)['reviews'],[])
        self.assertIsNone(self.row(changed)['appraisal']['strength'])
        self.data['patient_id']='another-synthetic';self.path.write_text(json.dumps(self.data))
        self.session.import_handoff(self.path);other=self.atlas.refresh()
        self.assertIsNone(self.row(other)['appraisal']['strength'])
        self.assertIsNone(other['previous_snapshot_id'])

    def test_export_preserves_source_and_adds_reviewed_order_without_overwrite(self):
        source=self.path.read_bytes();self.atlas.refresh();self.review(0,'support');result=self.atlas.refresh()
        exported=self.atlas.export(result,self.root/'atlas.json');raw=json.loads(exported.read_text())
        self.assertEqual([r['name'] for r in raw['risk_areas'][0]['investigational_evidence']],['Second','First','Third'])
        self.assertEqual(self.path.read_bytes(),source)
        with self.assertRaises(FileExistsError):self.atlas.export(result,self.path)
        self.assertEqual(self.path.read_bytes(),source)
        self.assertEqual(raw['feedback_update']['model_calls'],0)
        self.review(1,'oppose')
        with self.assertRaisesRegex(AtlasError,'changed'):self.atlas.export(result,self.root/'stale.json')
        self.assertFalse((self.root/'stale.json').exists())

    def test_missing_authorization_and_native_failure_never_publish_an_atlas(self):
        self.session.authentication=self.people[1]
        with self.assertRaises(AuthenticationError):self.atlas.refresh()
        self.session.import_handoff(self.path)
        class Failure:
            def run(self,context):raise AtlasError('Native service down')
        with self.assertRaises(AtlasError):ReviewedAtlas(self.session,Failure()).refresh()
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM reviewed_atlases').fetchone()[0],0)

    def test_review_arriving_during_tool_execution_rejects_stale_publication(self):
        parent=self
        class Concurrent(PolicyOnlyClient):
            def run(self,context):
                parent.review(0,'oppose')
                return super().run(context)
        with self.assertRaisesRegex(AtlasError,'Another review'):ReviewedAtlas(self.session,Concurrent()).refresh()
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM reviewed_atlases').fetchone()[0],0)


if __name__=='__main__':unittest.main()
