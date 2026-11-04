"""Immutable interpretations, saved drafts, export consent and ranked review views."""
from __future__ import annotations

import json
import time

from app.services.feedback_reason_schema import digest, validate_review_reason

SCHEMA = """
    CREATE TABLE IF NOT EXISTS feedback_analyses (
        id TEXT PRIMARY KEY, cache_key TEXT NOT NULL, result_json TEXT NOT NULL, created_epoch REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS feedback_analysis_cache ON feedback_analyses(cache_key,created_epoch);
    CREATE TABLE IF NOT EXISTS review_drafts (
        candidate_id TEXT NOT NULL REFERENCES candidates(id), version_id TEXT NOT NULL REFERENCES versions(id),
        clinician_id TEXT NOT NULL REFERENCES clinicians(id), decision TEXT NOT NULL,
        rationale TEXT NOT NULL, citation TEXT NOT NULL, updated_epoch REAL NOT NULL,
        PRIMARY KEY(candidate_id,version_id,clinician_id)
    );
    CREATE TABLE IF NOT EXISTS external_review_consents (
        clinician_id TEXT NOT NULL REFERENCES clinicians(id), dataset_id TEXT NOT NULL REFERENCES datasets(id),
        destination_key TEXT NOT NULL, destination_json TEXT NOT NULL, accepted_epoch REAL NOT NULL,
        PRIMARY KEY(clinician_id,dataset_id,destination_key)
    );
"""


def destination_key(identity):
    return digest({k:identity[k] for k in ('destination','provider','model')})


class ReasonedFeedback:
    def cached_analysis(self, key, ttl=86400):
        row=self.db.execute('SELECT * FROM feedback_analyses WHERE cache_key=? ORDER BY created_epoch DESC LIMIT 1',(key,)).fetchone()
        if row and time.time()-row['created_epoch']<=ttl:
            result=json.loads(row['result_json'])
            if digest([key,result])!=row['id']:
                raise ValueError('Saved Omega interpretation failed its integrity check; no score was changed.')
            return {'id':row['id'],'result':result,'cached':True}
        return None

    def save_analysis(self, key, result):
        identity=digest([key,result])
        with self.transaction():
            self.db.execute('INSERT OR IGNORE INTO feedback_analyses VALUES (?,?,?,?)',
                            (identity,key,json.dumps(result,sort_keys=True,ensure_ascii=False,allow_nan=False),time.time()))
        return {'id':identity,'result':result,'cached':False}

    def has_external_consent(self, clinician, identity):
        self.require_clinician(clinician)
        row=self.db.execute('SELECT 1 FROM external_review_consents WHERE clinician_id=? AND dataset_id=? AND destination_key=?',
                            (clinician.id,self.active_dataset(),destination_key(identity))).fetchone()
        return row is not None

    def record_external_consent(self, clinician, identity):
        self.require_clinician(clinician)
        public={k:identity[k] for k in ('destination','provider','model')}
        with self.transaction():
            self.db.execute('INSERT OR IGNORE INTO external_review_consents VALUES (?,?,?,?,?)',
                            (clinician.id,self.active_dataset(),destination_key(identity),json.dumps(public),time.time()))

    def save_draft(self, candidate, clinician, decision, rationale, citation, *, notes_only=False):
        self.require_clinician(clinician)
        if not notes_only:
            validate_review_reason(rationale)
        with self.transaction():
            if self.candidate(candidate['id'])['current_version']!=candidate['current_version']:
                raise ValueError('The evidence changed. Open the current version before reviewing.')
            self.db.execute('INSERT OR REPLACE INTO review_drafts VALUES (?,?,?,?,?,?,?)',
                            (candidate['id'],candidate['current_version'],clinician.id,decision,rationale,citation,time.time()))

    def review_draft(self, candidate, clinician):
        self.require_clinician(clinician)
        row=self.db.execute('SELECT * FROM review_drafts WHERE candidate_id=? AND version_id=? AND clinician_id=?',
                            (candidate['id'],candidate['current_version'],clinician.id)).fetchone()
        return dict(row) if row else None

    def discard_draft(self, candidate, clinician):
        self.require_clinician(clinician)
        with self.transaction():
            self.db.execute('DELETE FROM review_drafts WHERE candidate_id=? AND version_id=? AND clinician_id=?',
                            (candidate['id'],candidate['current_version'],clinician.id))

    def rankings(self, condition=''):
        groups={}
        for candidate in self.candidates():
            if condition and condition.casefold() not in candidate['condition'].casefold():
                continue
            group=groups.setdefault(candidate['condition'],{'condition':candidate['condition'],'ranked':[],'unranked':[]})
            assessment=self.belief_assessment(candidate)
            if not assessment or assessment['strength'] is None:
                group['unranked'].append(candidate)
            else:
                group['ranked'].append({'candidate':candidate,'assessment':assessment})
        for group in groups.values():
            group['ranked'].sort(key=lambda row:(-round(row['assessment']['strength'],10),-round(row['assessment']['confidence'],10),row['candidate']['name'].casefold(),row['candidate']['id']))
            previous=None
            rank=0
            for position,row in enumerate(group['ranked'],1):
                score=(round(row['assessment']['strength'],10),round(row['assessment']['confidence'],10))
                if score!=previous:rank=position
                row['rank']=rank
                previous=score
        return list(groups.values())
