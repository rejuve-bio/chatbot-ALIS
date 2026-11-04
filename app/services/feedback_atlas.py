"""Persisted feedback -> native Omega tools -> reviewed intervention atlas.

All data stays local. The symbolic workflow planner makes no LLM requests.
No service-wide patient match or cross-version review reuse is permitted.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess

from app.services.feedback_atlas_policy import POLICY, digest, assemble, changes
from app.services.feedback_auth import AuthenticationError

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / 'runtime/feedback/atlas'
SCHEMA = """
CREATE TABLE IF NOT EXISTS reviewed_atlases (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 dataset_id TEXT NOT NULL REFERENCES datasets(id), scope_key TEXT NOT NULL,
 input_digest TEXT NOT NULL, result_json TEXT NOT NULL, created_at TEXT NOT NULL,
 created_by TEXT NOT NULL REFERENCES clinicians(id)
);
CREATE INDEX IF NOT EXISTS reviewed_atlas_scope ON reviewed_atlases(dataset_id,scope_key,id);
"""


class AtlasError(RuntimeError):
    pass


class NativeAtlasClient:
    """Run the unmodified Core loop with a local symbolic workflow planner.

    The provider API is used by Omega for command dispatch, not a remote model.
    Native PLN results were already computed when the review was saved.
    """
    def __init__(self, container=None, timeout=30):
        self.container = container or os.getenv('OMEGA_FEEDBACK_CONTAINER', 'chatbot-alis-omega-mapping-1')
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}', self.container):
            raise ValueError('Invalid Omega container name.')
        self.timeout = timeout

    def run(self, context):
        names=('run.metta','skills.metta','channels.metta','memory.metta','bridge.py','atlas_planner.py')
        files = {name: (PROFILE/name).read_text() for name in names}
        files['atlas_policy.py'] = (ROOT/'app/services/feedback_atlas_policy.py').read_text().replace(
            'from app.services.feedback_triage import review_status, ORDER', 'from triage import review_status, ORDER')
        files['triage.py'] = (ROOT/'app/services/feedback_triage.py').read_text()
        manifest = json.loads((ROOT/'runtime/feedback/manifest.json').read_text())
        package = dict(context=context, files=files, expected_core=manifest['omega_commit'],
                       expected_petta=manifest['petta_commit'], timeout=self.timeout)
        try:
            p = subprocess.run(['docker', 'exec', '-i', self.container, 'python3', '-B', '-c',
                                (PROFILE/'runner.py').read_text()], input=json.dumps(package),
                               capture_output=True, text=True, timeout=self.timeout+10)
            value = json.loads(p.stdout)
            if p.returncode or value.get('error'):
                raise AtlasError(value.get('error', 'Native Omega atlas tools did not finish.'))
            expected = assemble(context)
            if (value['atlas'] != expected or value['input_digest'] != digest(context)
                    or value['model_calls'] != 0 or value['runtime']['core_commit'] != manifest['omega_commit']
                    or value['runtime']['petta_commit'] != manifest['petta_commit']
                    or value['runtime']['profile_sha256'] != digest(files)
                    or [t['tool'] for t in value['tool_trace']] != ['read-current-feedback', 'apply-review-policy', 'finish-reviewed-atlas']):
                raise AtlasError('Native Omega atlas output failed verification; no replacement atlas was saved.')
            return value
        except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
            raise AtlasError('Native Omega atlas assembly unavailable. Start '+self.container+
                             '; your review remains saved. No fallback atlas or model request was made.') from error


def context_for(store, minimum_decisive=0):
    if type(minimum_decisive) is not int or minimum_decisive < 0:
        raise ValueError('Minimum decisive reviews must be a nonnegative whole number.')
    entries = []
    original=json.loads(store.db.execute('SELECT original_json FROM datasets WHERE id=?',(store.active_dataset(),)).fetchone()[0])
    area_by_position={};position=0
    for area_index,area in enumerate(original['risk_areas']):
        for category in ('treatments','investigational_evidence','evidence'):
            for item in area.get(category,[]):
                position+=1;area_by_position[position]=area_index
    for c in store.candidates():
        a = store.belief_assessment(c)
        if a is None:
            raise AtlasError('The current evidence version has no saved appraisal. Reopen the handoff.')
        reviews = [{k: r[k] for k in ('id','clinician_id','reviewer','decision','rationale','citation',
                                     'created_at','concern_kind','concern_text')}
                   for r in store.current_reviews(c)]
        events = [dict(id=r['id'], reviewer=r['reviewer'], decision=r['decision'],
                       version_id=r['version_id'], supersedes_id=r['supersedes_id'],
                       note=r['rationale'], concern_kind=r['concern_kind'], concern_text=r['concern_text'],
                       created_at=r['created_at'])
                  for r in store.db.execute('SELECT * FROM reviews WHERE candidate_id=? ORDER BY id', (c['id'],))]
        native = a.get('native') or {}
        entries.append(dict(id=c['id'], entry_number=c['position'], area_index=area_by_position[c['position']], name=c['name'], condition=c['condition'],
                            category=c['category'], version_id=c['current_version'], source_version=c['source_version'],
                            statement=c['proposition'], data=c['data'], reviews=reviews, events=events,
                            event_head=events[-1]['id'] if events else 0,
                            appraisal={k:a.get(k) for k in ('strength','confidence','support','oppose','uncertain',
                                                          'assessment_id','adapter','confidence_scope','status')},
                            pln=dict(used=a['pln_used'], rule='Truth__Revision / Truth__w2c',
                                     program_sha256=native.get('program_sha256'), stdout=native.get('stdout'),
                                     omega_commit=native.get('omega_commit'), petta_commit=native.get('petta_commit'))))
    return dict(policy=POLICY, dataset_id=store.active_dataset(), minimum_decisive=minimum_decisive, entries=entries)


class ReviewedAtlas:
    def __init__(self, session, client=None):
        self.s = session
        self.store = session.store
        self.client = client or NativeAtlasClient()

    def _scope(self, minimum):
        # Source hashes also invalidate an atlas when its routing implementation changes.
        return digest([POLICY, minimum,
                       (ROOT/'app/services/feedback_atlas_policy.py').read_text(),
                       (ROOT/'app/services/feedback_triage.py').read_text()])

    def latest(self, minimum=0):
        row = self.store.db.execute('SELECT * FROM reviewed_atlases WHERE dataset_id=? AND scope_key=? ORDER BY id DESC LIMIT 1',
                                    (self.store.active_dataset(), self._scope(minimum))).fetchone()
        return dict(json.loads(row['result_json']), snapshot_id=row['id'], created_at=row['created_at']) if row else None

    def refresh(self, minimum=0, progress=None):
        authorized_dataset = self.store.active_dataset()
        self.s.require_access()
        with self.store.transaction():
            if self.store.active_dataset() != authorized_dataset:
                raise AtlasError('The opened file changed during authorization. Reopen your selected handoff.')
            context = context_for(self.store, minimum)
            previous = self.latest(minimum)
        key = digest(context)
        if previous and previous['input_digest'] == key:
            return dict(previous, rebuilt=False)
        if progress:progress('Omega is reading current reviews and assembling the reviewed atlas locally…')
        native = self.client.run(context)
        self.s.require_access()
        from app.services.feedback_store import now, encoded
        with self.store.transaction():
            # Do not publish a successful-looking stale result if another review arrived.
            if digest(context_for(self.store, minimum)) != key:
                raise AtlasError('Another review arrived during assembly. Your review is saved; refresh the atlas to include the latest responses.')
            current_previous = self.latest(minimum)
            if current_previous and current_previous['input_digest'] == key:
                return dict(current_previous, rebuilt=False)
            atlas = native['atlas']
            result = dict(input_digest=key, atlas=atlas, changes=changes(current_previous['atlas'] if current_previous else None, atlas, context),
                          tool_trace=native['tool_trace'], runtime=native['runtime'], model_calls=0,
                          previous_snapshot_id=current_previous['snapshot_id'] if current_previous else None)
            created = now()
            cursor = self.store.db.execute('INSERT INTO reviewed_atlases(dataset_id,scope_key,input_digest,result_json,created_at,created_by) VALUES (?,?,?,?,?,?)',
                        (context['dataset_id'], self._scope(minimum), key, encoded(result), created, self.s.reviewer.id))
        return dict(result, snapshot_id=cursor.lastrowid, created_at=created, rebuilt=True)

    def export(self, snapshot, path):
        self.s.require_access()
        if snapshot['atlas']['dataset_id'] != self.store.active_dataset():
            raise AuthenticationError('This atlas belongs to a different handoff.')
        with self.store.transaction():
            if snapshot['atlas']['dataset_id'] != self.store.active_dataset():
                raise AuthenticationError('The opened handoff changed before export.')
            if digest(context_for(self.store, snapshot['atlas']['minimum_decisive'])) != snapshot['input_digest']:
                raise AtlasError('Feedback has changed. Refresh the reviewed atlas before exporting.')
            raw = json.loads(self.store.db.execute('SELECT original_json FROM datasets WHERE id=?', (self.store.active_dataset(),)).fetchone()[0])
        rows = snapshot['atlas']['entries']
        for area_index,area in enumerate(raw['risk_areas']):
            for category in ('treatments','investigational_evidence','evidence'):
                matches = [r for r in rows if r['area_index'] == area_index and r['category'] == category]
                if matches:
                    area[category] = [dict(r['data'], clinician_feedback={k:r[k] for k in ('id','version_id','statement','research_order','routing','appraisal','reviews','pln')}) for r in matches]
        raw['feedback_update'] = dict(snapshot_id=snapshot['snapshot_id'], created_at=snapshot['created_at'],
                                     input_digest=snapshot['input_digest'], policy=POLICY, changes=snapshot['changes'],
                                     runtime=snapshot['runtime'], tool_trace=snapshot['tool_trace'],
                                     purpose='Clinician-reviewed research atlas', model_calls=0)
        path = Path(path).expanduser().resolve()
        # Exclusive creation protects the supplied handoff and prior exports.
        fd = os.open(str(path), os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as output:
            json.dump(raw, output, ensure_ascii=False, indent=2, allow_nan=False)
        return path
