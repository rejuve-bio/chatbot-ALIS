"""Versioned source facts and evidence calculations, separate from opinions.

One current contribution per source/effect, irrespective of how many clinicians
have read it. Confirmations update facts, never add a clinician as study data.
"""
from __future__ import annotations
import json
import copy
from app.services.feedback_source_schema import validate_source_assessments, source_state_key
from app.services.feedback_source_scores import score_sources, digest, canonical

SCHEMA='''
CREATE TABLE IF NOT EXISTS source_evidence_events (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 candidate_id TEXT NOT NULL REFERENCES candidates(id),
 version_id TEXT NOT NULL REFERENCES versions(id),
 clinician_id TEXT NOT NULL REFERENCES clinicians(id),
 analysis_id TEXT REFERENCES feedback_analyses(id),
 records_json TEXT NOT NULL, input_digest TEXT NOT NULL,
 trigger TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_evidence_scores (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 event_id INTEGER NOT NULL REFERENCES source_evidence_events(id),
 candidate_id TEXT NOT NULL REFERENCES candidates(id),
 version_id TEXT NOT NULL REFERENCES versions(id),
 result_json TEXT NOT NULL, created_at TEXT NOT NULL
);
'''


class SourceEvidence:
    def evidence_events(self,candidate):
        rows=self.db.execute('SELECT e.*,c.name AS reviewer FROM source_evidence_events e JOIN clinicians c ON c.id=e.clinician_id '
            'WHERE candidate_id=? ORDER BY e.id',(candidate['id'],)).fetchall()
        return [dict(r,records=json.loads(r['records_json'])) for r in rows]

    def source_records(self,candidate):
        row=self.db.execute('SELECT records_json FROM source_evidence_events WHERE candidate_id=? AND version_id=? ORDER BY id DESC LIMIT 1',
            (candidate['id'],candidate['current_version'])).fetchone()
        return json.loads(row[0]) if row else []

    def latest_evidence_score(self,candidate):
        row=self.db.execute('SELECT * FROM source_evidence_scores WHERE candidate_id=? AND version_id=? ORDER BY id DESC LIMIT 1',
            (candidate['id'],candidate['current_version'])).fetchone()
        if row is None:return None
        return dict(json.loads(row['result_json']),assessment_id=row['id'],event_id=row['event_id'],created_at=row['created_at'])

    def source_plan(self,candidate,result):
        validate_source_assessments(result['analysis'].get('source_assessments'),result['context'])
        sources={s['id']:s for s in result['context']['sources']}
        # Sources omitted by the bounded inspection remain recorded. A new
        # reviewer neither duplicates them nor silently erases them.
        records={r['source']['id']:r for r in self.source_records(candidate)}
        for facts in result['analysis']['source_assessments']:
            records[facts['source_id']]={'facts':facts,'source':sources[facts['source_id']],'status':'active'}
        return [records[k] for k in sorted(records)]

    @staticmethod
    def evidence_digest(records):
        return digest([{'source_id':r['source']['id'],'text_sha256':r['source']['text_sha256'],
                        'facts':source_state_key(r['facts']),'status':r['status'],
                        'independence_confirmed':bool(r.get('independence_confirmed',False))} for r in records])

    def _apply_source_plan(self,candidate,reviewer,records,analysis_id,trigger):
        """Called within the caller's transaction; errors roll back the review too."""
        from app.services.feedback_store import now
        old=self.source_records(candidate)
        key=self.evidence_digest(records)
        if self.evidence_digest(old)==key:
            return self.latest_evidence_score(candidate),False
        result=score_sources(records)
        # Native calculation is required when supported; no fallback numerical
        # score is saved on a runtime or integrity error.
        created=now()
        cursor=self.db.execute('INSERT INTO source_evidence_events(candidate_id,version_id,clinician_id,analysis_id,records_json,input_digest,trigger,created_at) VALUES (?,?,?,?,?,?,?,?)',
            (candidate['id'],candidate['current_version'],reviewer.id,analysis_id,canonical(records),key,trigger,created))
        event=cursor.lastrowid
        result.update(input_digest=key,trigger=trigger)
        score=self.db.execute('INSERT INTO source_evidence_scores(event_id,candidate_id,version_id,result_json,created_at) VALUES (?,?,?,?,?)',
            (event,candidate['id'],candidate['current_version'],canonical(result),created))
        self._rebuild_belief(candidate, 'Source provenance updated; rebuild appraisal groups')
        return dict(result,assessment_id=score.lastrowid,event_id=event,created_at=created),True

    def confirm_source_analysis(self,candidate,reviewer,analysis_id,expected_digest=None):
        self.require_clinician(reviewer)
        with self.transaction():
            current=self.candidate(candidate['id'])
            if candidate['current_version']!=current['current_version']:raise ValueError('Evidence changed; open the current version before confirming.')
            if expected_digest is not None and self.evidence_digest(self.source_records(current))!=expected_digest:
                raise ValueError('Another reviewer changed the source facts. Inspect the new evidence before confirming.')
            row=self.db.execute('SELECT result_json FROM feedback_analyses WHERE id=?',(analysis_id,)).fetchone()
            if row is None:raise ValueError('Unknown source assessment.')
            result=json.loads(row[0])
            from app.services.feedback_omega import packet
            actual=dict(result['context']);actual.pop('sources',None)
            review=actual['review']
            if actual!=packet(current,review['decision'],review['reason'],review['citation'],clinician=reviewer):
                raise ValueError('The source assessment belongs to another claim or evidence version.')
            before=self.latest_evidence_score(current)
            after,changed=self._apply_source_plan(current,reviewer,self.source_plan(current,result),analysis_id,'Confirmed source assessment by '+reviewer.name)
        return {'before':before,'after':after,'changed':changed}

    def restore_source_event(self,candidate,reviewer,event_id,reason):
        self.require_clinician(reviewer)
        if not reason.strip():raise ValueError('A reason is required for an evidence rollback.')
        with self.transaction():
            current=self.candidate(candidate['id'])
            if current['current_version']!=candidate['current_version']:raise ValueError('Evidence changed; reopen it first.')
            row=self.db.execute('SELECT * FROM source_evidence_events WHERE id=? AND candidate_id=? AND version_id=?',
                (event_id,candidate['id'],candidate['current_version'])).fetchone()
            if row is None:raise ValueError('Choose a source event for this evidence version from history N.')
            before=self.latest_evidence_score(current)
            after,changed=self._apply_source_plan(current,reviewer,json.loads(row['records_json']),row['analysis_id'],
                'Restore source event #'+str(event_id)+' by '+reviewer.name+': '+reason.strip())
        return {'before':before,'after':after,'changed':changed}

    def edited_source_plan(self,candidate,source_id,field,value,quote):
        records=copy.deepcopy(self.source_records(candidate))
        record=next((r for r in records if r['source']['id']==source_id),None)
        if record is None:raise ValueError('Choose a source ID from evidence N.')
        if not quote.strip() or quote not in record['source']['text']:
            raise ValueError('The correction needs an exact quote from the saved source text. Use source-text N PMID:ID to inspect it.')
        allowed={'population_type','design','comparison','condition_match','intervention_match','outcome_role','population','comparator','outcome',
                 'effect.measure','effect.estimate','effect.ci_lower','effect.ci_upper','effect.ci_level','effect.direction','effect.quote','exclusion_reason'}
        if field not in allowed:raise ValueError('Supported fields: '+', '.join(sorted(allowed)))
        if field.startswith('effect.'):
            if record['facts']['effect'] is None:raise ValueError('No effect tuple exists. Ask Omega to inspect a cited result before correcting its individual values.')
            record['facts']['effect'][field.split('.')[1]]=value
        else:record['facts'][field]=value
        quotes=record['facts']['source_quotes']
        if quote not in quotes:record['facts']['source_quotes']=(quotes[:2]+[quote])
        validate_source_assessments([r['facts'] for r in records],{'sources':[r['source'] for r in records]})
        return records

    def correct_source(self,candidate,reviewer,source_id,field,value,reason,quote,expected_digest=None):
        self.require_clinician(reviewer)
        if not reason.strip():raise ValueError('A reason is required for the source correction.')
        with self.transaction():
            current=self.candidate(candidate['id'])
            if current['current_version']!=candidate['current_version']:raise ValueError('The handoff changed; reopen it before correcting.')
            if expected_digest is not None and self.evidence_digest(self.source_records(current))!=expected_digest:
                raise ValueError('Another reviewer changed the source facts. Inspect the new evidence before confirming.')
            records=self.edited_source_plan(current,source_id,field,value,quote)
            before=self.latest_evidence_score(current)
            after,changed=self._apply_source_plan(current,reviewer,records,None,
                source_id+' '+field+' corrected by '+reviewer.name+': '+reason.strip()+' | quote: '+quote)
        return {'before':before,'after':after,'changed':changed}
