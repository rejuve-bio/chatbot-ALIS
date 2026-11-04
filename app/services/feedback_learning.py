"""Source-matched reuse of explicitly approved evidence corrections.

This is knowledge acquisition, not numerical parameter training. Review votes
remain patient-specific. Reuse requires two authenticated people, an explicit
shared scope, and the same intervention, condition and upstream source snapshot.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re

from app.services.feedback_pln import assess

REUSABLE_FIELDS = {"why", "note", "review_claim"}


def fingerprint(value):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def claim_key(candidate):
    data = candidate["data"]
    drugs = data.get("drugs") or []
    return fingerprint([candidate["condition"].strip().casefold(), candidate["category"],
                        candidate["name"].strip().casefold(), sorted(str(d).strip().casefold() for d in drugs)])


def public_reference(value):
    return bool(re.search(r"\bPMID\s*:?\s*\d+\b|\bNCT\d{8}\b|\b10\.\d{4,9}/\S+|https://\S+",
                          value, flags=re.IGNORECASE))


def timestamp():
    return datetime.now(timezone.utc).isoformat()


SCHEMA = """
    CREATE TABLE IF NOT EXISTS learned_knowledge (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        claim_key TEXT NOT NULL, source_hash TEXT NOT NULL,
        intervention TEXT NOT NULL, condition_name TEXT NOT NULL,
        field TEXT NOT NULL, original_json TEXT NOT NULL, replacement TEXT NOT NULL,
        rationale TEXT NOT NULL, citation TEXT NOT NULL,
        correction_id INTEGER NOT NULL UNIQUE REFERENCES corrections(id),
        proposer_id TEXT NOT NULL REFERENCES clinicians(id),
        approver_id TEXT NOT NULL REFERENCES clinicians(id),
        approval_reason TEXT NOT NULL, created_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','superseded','retired')),
        replaces_id INTEGER REFERENCES learned_knowledge(id)
    );
    CREATE UNIQUE INDEX IF NOT EXISTS active_learned_claim
        ON learned_knowledge(claim_key,source_hash,field) WHERE status='active';
    CREATE TABLE IF NOT EXISTS knowledge_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        knowledge_id INTEGER NOT NULL REFERENCES learned_knowledge(id),
        action TEXT NOT NULL, clinician_id TEXT NOT NULL REFERENCES clinicians(id),
        reason TEXT NOT NULL, created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS knowledge_applications (
        knowledge_id INTEGER NOT NULL REFERENCES learned_knowledge(id),
        candidate_id TEXT NOT NULL REFERENCES candidates(id),
        version_id TEXT NOT NULL REFERENCES versions(id), applied_at TEXT NOT NULL,
        PRIMARY KEY(knowledge_id,version_id)
    );
"""


class KnowledgeMemory:
    def _source_data(self, candidate):
        row = self.db.execute("SELECT data_json FROM versions WHERE id=?", (candidate["source_version"],)).fetchone()
        return json.loads(row[0])

    def _publish_knowledge(self, correction, candidate, approver, reason):
        source = self._source_data(candidate)
        key, source_hash = claim_key(candidate), fingerprint(source)
        previous = self.db.execute("SELECT id FROM learned_knowledge WHERE claim_key=? AND source_hash=? AND field=? AND status='active'",
                                   (key, source_hash, correction["field"])).fetchone()
        if previous:
            self.db.execute("UPDATE learned_knowledge SET status='superseded' WHERE id=?", (previous[0],))
            self.db.execute("INSERT INTO knowledge_events(knowledge_id,action,clinician_id,reason,created_at) VALUES (?,?,?,?,?)",
                            (previous[0], "superseded", approver.id, reason, timestamp()))
        cursor = self.db.execute(
            "INSERT INTO learned_knowledge(claim_key,source_hash,intervention,condition_name,field,original_json,replacement,"
            "rationale,citation,correction_id,proposer_id,approver_id,approval_reason,created_at,replaces_id,grounding_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (key, source_hash, candidate["name"], candidate["condition"], correction["field"],
             json.dumps(source.get(correction["field"]), ensure_ascii=False), correction["replacement"],
             correction["rationale"], correction["citation"], correction["id"], correction["proposer_clinician_id"],
             approver.id, reason, timestamp(), previous[0] if previous else None, correction["grounding_json"]))
        identity = cursor.lastrowid
        self.db.execute("INSERT INTO knowledge_events(knowledge_id,action,clinician_id,reason,created_at) VALUES (?,?,?,?,?)",
                        (identity, "approved", approver.id, reason, timestamp()))
        return identity

    def _apply_knowledge(self, candidate):
        # A local correction can contain patient-specific facts. Preserve it;
        # never replace it with a shared evidence statement automatically.
        if candidate["origin"] != "upstream" and not candidate["origin"].startswith(("learned knowledge", "knowledge removed")):
            return False
        source = self._source_data(candidate)
        rows = self.db.execute("SELECT * FROM learned_knowledge WHERE claim_key=? AND source_hash=? AND status='active' ORDER BY id",
                               (claim_key(candidate), fingerprint(source))).fetchall()
        if not rows and not candidate["origin"].startswith("learned knowledge"):
            return False
        data = dict(source)
        for row in rows:
            data[row["field"]] = row["replacement"]
            if row['field']=='review_claim' and row['grounding_json']:
                data['review_claim_grounding']=json.loads(row['grounding_json'])
        identities = [row["id"] for row in rows]
        version = fingerprint([candidate["id"], candidate["source_version"], "learned knowledge", identities]) if rows else fingerprint(
            [candidate["id"], candidate["source_version"], "knowledge removed", candidate["current_version"]])
        if version == candidate["current_version"]:
            return False
        origin = "learned knowledge " + ", ".join("#" + str(i) for i in identities) if rows else "knowledge removed; source restored"
        self.db.execute("INSERT OR IGNORE INTO versions VALUES (?,?,?,?,?,?)",
                        (version, candidate["id"], json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
                         origin, candidate["current_version"], timestamp()))
        self.db.execute("UPDATE candidates SET current_version=? WHERE id=?", (version, candidate["id"]))
        for identity in identities:
            self.db.execute("INSERT OR IGNORE INTO knowledge_applications VALUES (?,?,?,?)",
                            (identity, candidate["id"], version, timestamp()))
        updated = self.candidate(candidate["id"])
        if self.latest_assessment(updated) is None:
            self._save_assessment(updated, assess([]), origin + "; new version needs its own clinician review")
        self._rebuild_belief(updated, origin + "; current claim version")
        return True

    def _refresh_known_candidates(self):
        count = 0
        for row in self.db.execute("SELECT id FROM candidates").fetchall():
            count += self._apply_knowledge(self.candidate(row[0]))
        return count

    def applied_knowledge(self, candidate):
        return [dict(row) for row in self.db.execute(
            "SELECT k.id,k.field,k.replacement,k.citation,k.rationale,k.approval_reason,k.created_at,k.status,"
            "k.proposer_id,k.approver_id,k.grounding_json,p.name AS proposer,a.name AS approver "
            "FROM knowledge_applications u JOIN learned_knowledge k ON k.id=u.knowledge_id "
            "JOIN clinicians p ON p.id=k.proposer_id JOIN clinicians a ON a.id=k.approver_id "
            "WHERE u.version_id=? ORDER BY k.id", (candidate["current_version"],))]

    def learning_details(self, candidate):
        current = {row["id"] for row in self.applied_knowledge(candidate)}
        source_hash = fingerprint(self._source_data(candidate))
        result = []
        for row in self.db.execute(
            "SELECT k.*,p.name AS proposer,a.name AS approver FROM learned_knowledge k "
            "JOIN clinicians p ON p.id=k.proposer_id JOIN clinicians a ON a.id=k.approver_id "
            "WHERE k.claim_key=? ORDER BY k.id", (claim_key(candidate),)):
            item = dict(row)
            item["source_matches"] = item["source_hash"] == source_hash
            item["applied"] = item["id"] in current
            item["events"] = [dict(e) for e in self.db.execute(
                "SELECT action,clinician_id,reason,created_at FROM knowledge_events WHERE knowledge_id=? ORDER BY id", (item["id"],))]
            # The origin correction and its patient record stay private to their
            # handoff. This view exposes the explicitly shared statement only.
            item.pop("correction_id")
            item.pop("original_json")
            result.append(item)
        return result

    def retire_knowledge(self, identity, reviewer, reason):
        self.require_clinician(reviewer)
        if not reason.strip():
            raise ValueError("Give a reason to stop reusing this knowledge.")
        with self.transaction():
            row = self.db.execute("SELECT * FROM learned_knowledge WHERE id=?", (identity,)).fetchone()
            if row is None or row["status"] != "active":
                raise ValueError("Choose an active knowledge ID from learning N.")
            if reviewer.id != row["approver_id"]:
                raise ValueError("The clinician who approved this knowledge must retire it.")
            self.db.execute("UPDATE learned_knowledge SET status='retired' WHERE id=?", (identity,))
            self.db.execute("INSERT INTO knowledge_events(knowledge_id,action,clinician_id,reason,created_at) VALUES (?,?,?,?,?)",
                            (identity, "retired", reviewer.id, reason.strip(), timestamp()))
            count = self._refresh_known_candidates()
        return count
