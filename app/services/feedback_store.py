"""Local, versioned clinician-review store for imported intervention assessments."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3

from app.services.feedback_pln import assess
from app.services.feedback_beliefs import AppraisalBeliefs, SCHEMA as BELIEF_SCHEMA
from app.services.feedback_auth import AccountAlreadyExists, AuthenticationError, Clinician
from app.services.feedback_learning import KnowledgeMemory, REUSABLE_FIELDS, SCHEMA as KNOWLEDGE_SCHEMA, public_reference
from app.services.feedback_review import ReasonedFeedback, SCHEMA as REVIEW_SCHEMA
from app.services.feedback_evidence import SourceEvidence, SCHEMA as SOURCE_EVIDENCE_SCHEMA
from app.services.feedback_claims import ClaimReview, SCHEMA as CLAIM_SCHEMA
from app.services.feedback_triage import CONCERNS
from app.services.feedback_atlas import SCHEMA as ATLAS_SCHEMA

DECISIONS = {"support", "oppose", "uncertain", "withdrawn"}
CORRECTION_FIELDS = {"why", "note", "warning", "evidence", "evidence_tier", "review_claim"}


def encoded(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def candidates_from_json(data: dict) -> list[dict]:
    if not isinstance(data, dict) or not isinstance(data.get("risk_areas"), list):
        raise ValueError("Expected an intervention JSON object with a risk_areas list.")
    patient = data.get("patient_id")
    if not isinstance(patient, str) or not patient.strip():
        raise ValueError("The handoff must include patient_id to scope reviews correctly.")
    result, seen = [], set()
    for area in data["risk_areas"]:
        if not isinstance(area, dict) or not isinstance(area.get("disease_name"), str) or not area["disease_name"].strip():
            raise ValueError("Every risk area needs a disease_name.")
        condition = area["disease_name"].strip()
        for category in ("treatments", "investigational_evidence", "evidence"):
            entries = area.get(category, [])
            if not isinstance(entries, list):
                raise ValueError(category + " must be a list.")
            for item in entries:
                if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
                    raise ValueError("Every intervention entry needs a name.")
                name = item["name"].strip()
                upstream = item.get("id") or item.get("intervention_id")
                identity = [patient, condition.casefold(), category,
                            str(upstream) if upstream else name.casefold()]
                key = digest(identity)
                if key in seen:
                    raise ValueError("Duplicate intervention identity in one handoff: " + name)
                seen.add(key)
                result.append({"id": key, "patient_id": patient, "condition": condition,
                               "category": category, "name": name, "data": item})
    if not result:
        raise ValueError("No intervention entries were found in this JSON.")
    return result


class FeedbackStore(KnowledgeMemory, ReasonedFeedback, SourceEvidence, AppraisalBeliefs, ClaimReview):
    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        existed = self.path.parent.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not existed:
            self.path.parent.chmod(0o700)
        self.db = sqlite3.connect(str(self.path), isolation_level=None, timeout=10)
        self.path.chmod(0o600)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        self.belief_policy_migrated = version < 10
        if version not in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13):
            raise ValueError("This feedback database has an unsupported schema version.")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS clinicians (
                id TEXT PRIMARY KEY, provider TEXT NOT NULL, subject TEXT NOT NULL,
                issuer TEXT NOT NULL, name TEXT NOT NULL, email TEXT NOT NULL,
                clinic_id TEXT, password_hash TEXT, is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL, last_login_at TEXT,
                UNIQUE(provider,issuer,subject), UNIQUE(provider,issuer,email)
            );
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY, source_name TEXT NOT NULL, original_json TEXT NOT NULL,
                imported_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS file_review_access (
                dataset_id TEXT NOT NULL REFERENCES datasets(id),
                clinician_id TEXT NOT NULL REFERENCES clinicians(id),
                source_name TEXT NOT NULL, granted_at TEXT NOT NULL,
                PRIMARY KEY(dataset_id, clinician_id)
            );
            CREATE TABLE IF NOT EXISTS candidates (
                id TEXT PRIMARY KEY, patient_id TEXT NOT NULL, condition TEXT NOT NULL,
                category TEXT NOT NULL, name TEXT NOT NULL, source_version TEXT NOT NULL,
                current_version TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS versions (
                id TEXT PRIMARY KEY, candidate_id TEXT NOT NULL REFERENCES candidates(id),
                data_json TEXT NOT NULL, origin TEXT NOT NULL, parent_id TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dataset_entries (
                dataset_id TEXT NOT NULL REFERENCES datasets(id), position INTEGER NOT NULL,
                candidate_id TEXT NOT NULL REFERENCES candidates(id),
                PRIMARY KEY(dataset_id, position), UNIQUE(dataset_id, candidate_id)
            );
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL REFERENCES candidates(id),
                version_id TEXT NOT NULL REFERENCES versions(id),
                reviewer TEXT NOT NULL, reviewer_key TEXT NOT NULL,
                clinician_id TEXT REFERENCES clinicians(id),
                decision TEXT NOT NULL CHECK(decision IN ('support','oppose','uncertain','withdrawn')),
                rationale TEXT NOT NULL, citation TEXT NOT NULL,
                supersedes_id INTEGER REFERENCES reviews(id), created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS review_identity ON reviews(candidate_id, version_id, reviewer_key, id);
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL REFERENCES candidates(id),
                version_id TEXT NOT NULL REFERENCES versions(id),
                result_json TEXT NOT NULL, trigger TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL REFERENCES candidates(id),
                base_version TEXT NOT NULL REFERENCES versions(id), field TEXT NOT NULL,
                replacement TEXT NOT NULL, rationale TEXT NOT NULL, citation TEXT NOT NULL,
                proposer TEXT NOT NULL, proposer_key TEXT NOT NULL,
                proposer_clinician_id TEXT REFERENCES clinicians(id),
                resolver_clinician_id TEXT REFERENCES clinicians(id),
                status TEXT NOT NULL DEFAULT 'pending', resolver TEXT, resolution_reason TEXT,
                created_at TEXT NOT NULL, resolved_at TEXT
            );
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        self.db.executescript(KNOWLEDGE_SCHEMA)
        self.db.executescript(REVIEW_SCHEMA)
        self.db.executescript(SOURCE_EVIDENCE_SCHEMA)
        self.db.executescript(BELIEF_SCHEMA)
        self.db.executescript(CLAIM_SCHEMA)
        self.db.executescript(ATLAS_SCHEMA)
        with self.transaction():
            for table, column in (("reviews", "clinician_id"), ("corrections", "proposer_clinician_id"),
                                  ("corrections", "resolver_clinician_id")):
                columns = {row["name"] for row in self.db.execute("PRAGMA table_info(" + table + ")")}
                if column not in columns:
                    self.db.execute("ALTER TABLE " + table + " ADD COLUMN " + column + " TEXT REFERENCES clinicians(id)")
            correction_columns = {row["name"] for row in self.db.execute("PRAGMA table_info(corrections)")}
            if "reusable" not in correction_columns:
                self.db.execute("ALTER TABLE corrections ADD COLUMN reusable INTEGER NOT NULL DEFAULT 0")
            for table in ('corrections','learned_knowledge'):
                columns={row['name'] for row in self.db.execute('PRAGMA table_info('+table+')')}
                if 'grounding_json' not in columns:
                    self.db.execute('ALTER TABLE '+table+' ADD COLUMN grounding_json TEXT')
            review_columns = {row["name"] for row in self.db.execute("PRAGMA table_info(reviews)")}
            for column, definition in (("analysis_id", "TEXT REFERENCES feedback_analyses(id)"),
                                       ("analysis_confirmed", "INTEGER NOT NULL DEFAULT 0"), ("submitted_decision", "TEXT"),
                                       ("decision_override", "INTEGER NOT NULL DEFAULT 0"),
                                       ("reason_mode", "TEXT NOT NULL DEFAULT 'legacy'"),
                                       ("concern_kind", "TEXT NOT NULL DEFAULT ''"),
                                       ("concern_text", "TEXT NOT NULL DEFAULT ''")):
                if column not in review_columns:
                    self.db.execute("ALTER TABLE reviews ADD COLUMN " + column + " " + definition)
            if version == 1:
                # Legacy names never become authenticated accounts by matching
                # display names. Preserve events and retract only their effect.
                self.db.execute("UPDATE corrections SET status='legacy_unverified' WHERE status='pending'")
                self.db.execute("UPDATE candidates SET current_version=source_version")
                for row in self.db.execute("SELECT id FROM candidates").fetchall():
                    candidate = self.candidate(row["id"])
                    self._save_assessment(candidate, assess([]), "Authentication enabled; legacy name-only reviews excluded")
            if version < 10:
                for row in self.db.execute("SELECT id FROM candidates").fetchall():
                    self._rebuild_belief(self.candidate(row["id"]), "Review-count policy enabled; original reviews and prior scores preserved")
            self.db.execute("PRAGMA user_version=13")

    def close(self):
        self.db.close()

    def reconnect(self):
        """Refresh an idle connection after external authentication/terminal I/O."""
        if self.db.in_transaction:
            raise RuntimeError("Cannot refresh the review connection inside a transaction.")
        self.db.close()
        self.db = sqlite3.connect(str(self.path), isolation_level=None, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")

    def bind_auth(self, provider: str, issuer: str):
        self.reconnect()
        binding = encoded([provider, issuer])
        with self.transaction():
            row = self.db.execute("SELECT value FROM settings WHERE key='auth_binding'").fetchone()
            if row and row[0] != binding:
                raise AuthenticationError("This database belongs to a different login provider. Use a separate --db file.")
            self.db.execute("INSERT OR IGNORE INTO settings VALUES ('auth_binding',?)", (binding,))

    def register_local_clinician(self, clinician: Clinician, password_hash: str):
        self.bind_auth("local", "local-feedback")
        try:
            with self.transaction():
                self.db.execute("INSERT INTO clinicians(id,provider,subject,issuer,name,email,password_hash,created_at) "
                                "VALUES (?,?,?,?,?,?,?,?)", (clinician.id, clinician.provider, clinician.subject,
                                clinician.issuer, clinician.name, clinician.email, password_hash, now()))
        except sqlite3.IntegrityError:
            raise AccountAlreadyExists("Account ALREADY EXISTS in this local database. No additional account was created; log in instead.") from None

    def local_account(self, email: str):
        row = self.db.execute("SELECT * FROM clinicians WHERE provider='local' AND issuer='local-feedback' AND email=?", (email,)).fetchone()
        return dict(row) if row else None

    def local_account_count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM clinicians WHERE provider='local' AND issuer='local-feedback'").fetchone()[0]

    def remember_alis_clinician(self, clinician: Clinician):
        self.bind_auth("alis", clinician.issuer)
        with self.transaction():
            self.db.execute("INSERT INTO clinicians(id,provider,subject,issuer,name,email,clinic_id,created_at,last_login_at) "
                            "VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,email=excluded.email,"
                            "clinic_id=excluded.clinic_id,is_active=1,last_login_at=excluded.last_login_at",
                            (clinician.id, clinician.provider, clinician.subject, clinician.issuer,
                             clinician.name, clinician.email, clinician.clinic_id, now(), now()))

    def record_login(self, clinician_id: str):
        with self.transaction():
            self.db.execute("UPDATE clinicians SET last_login_at=? WHERE id=?", (now(), clinician_id))

    def require_clinician(self, clinician: Clinician):
        if not isinstance(clinician, Clinician):
            raise AuthenticationError("An authenticated clinician account is required; a reviewer name is not an identity.")
        row = self.db.execute("SELECT * FROM clinicians WHERE id=?", (clinician.id,)).fetchone()
        if (row is None or not row["is_active"] or row["provider"] != clinician.provider
                or row["subject"] != clinician.subject or row["issuer"] != clinician.issuer):
            raise AuthenticationError("The clinician account is unavailable or inactive. Log in again.")

    def clinician_profile(self, clinician_id: str) -> dict:
        row = self.db.execute("SELECT id,provider,subject,issuer,name,email,clinic_id,created_at,last_login_at "
                              "FROM clinicians WHERE id=?", (clinician_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown clinician account.")
        return dict(row)

    @contextmanager
    def transaction(self):
        # A CLI can wait on terminal/network I/O for an arbitrary time between
        # commands. Start each write with a fresh idle connection, rather than
        # retaining a WAL handle across that wait. Never reopen mid-transaction.
        self.reconnect()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def import_file(self, path: Path, *, file_reviewer: Clinician | None = None) -> dict:
        path = path.expanduser().resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        return self.import_data(data, path.name, file_reviewer=file_reviewer)

    def import_data(self, data: dict, source_name: str, *, file_reviewer: Clinician | None = None) -> dict:
        """Import one read of a handoff; explicit file selection grants only its opener access."""
        entries = candidates_from_json(data)
        if file_reviewer is not None:
            self.require_clinician(file_reviewer)
        dataset_id = digest(data)
        created = changed = learned = 0
        with self.transaction():
            self.db.execute("INSERT OR IGNORE INTO datasets VALUES (?,?,?,?)",
                            (dataset_id, source_name, encoded(data), now()))
            for position, candidate in enumerate(entries, 1):
                key = candidate["id"]
                version = digest([key, candidate["data"]])
                old = self.db.execute("SELECT * FROM candidates WHERE id=?", (key,)).fetchone()
                if old is None:
                    created += 1
                    self.db.execute("INSERT INTO candidates VALUES (?,?,?,?,?,?,?)",
                                    (key, candidate["patient_id"], candidate["condition"], candidate["category"],
                                     candidate["name"], version, version))
                elif old["source_version"] != version:
                    changed += 1
                    self.db.execute("UPDATE candidates SET name=?,source_version=?,current_version=? WHERE id=?",
                                    (candidate["name"], version, version, key))
                self.db.execute("INSERT OR IGNORE INTO versions VALUES (?,?,?,?,?,?)",
                                (version, key, encoded(candidate["data"]), "upstream", None, now()))
                self.db.execute("INSERT OR IGNORE INTO dataset_entries VALUES (?,?,?)",
                                (dataset_id, position, key))
                # Reimporting identical source preserves accepted local corrections.
                current = self.candidate(key)
                self._apply_knowledge(current)
                current = self.candidate(key)
                learned += bool(current["learned_knowledge"])
                self._rebuild_belief(current, "Handoff opened; current evidence version")
                if self.latest_assessment(current) is None:
                    self._save_assessment(current, assess([]), "Initial import; no clinician judgments")
            self.db.execute("INSERT OR REPLACE INTO settings VALUES ('dataset',?)", (dataset_id,))
            if file_reviewer is not None:
                self.db.execute("INSERT OR IGNORE INTO file_review_access VALUES (?,?,?,?)",
                                (dataset_id, file_reviewer.id, source_name, now()))
        return {"id": dataset_id, "total": len(entries), "new": created, "changed": changed, "learned": learned}

    def require_file_review(self, clinician: Clinician, patient_id: str | None = None):
        self.require_clinician(clinician)
        dataset_id = self.active_dataset()
        access = self.db.execute("SELECT 1 FROM file_review_access WHERE dataset_id=? AND clinician_id=?",
                                 (dataset_id, clinician.id)).fetchone()
        if access is None:
            raise AuthenticationError(
                'Open the supplied JSON for this clinician first: import "FILE.json", or restart with --input FILE.json. '
                'File review access is recorded separately for each clinician.'
            )
        if patient_id is not None:
            row = self.db.execute("SELECT 1 FROM dataset_entries d JOIN candidates c ON c.id=d.candidate_id "
                                  "WHERE d.dataset_id=? AND c.patient_id=? LIMIT 1", (dataset_id, patient_id)).fetchone()
            if row is None:
                raise AuthenticationError("This record is outside the opened review file.")

    def require_active_candidate(self, candidate_id: str):
        row = self.db.execute("SELECT 1 FROM dataset_entries WHERE dataset_id=? AND candidate_id=?",
                              (self.active_dataset(), candidate_id)).fetchone()
        if row is None:
            raise AuthenticationError("This record is outside the opened review file.")

    def active_dataset(self) -> str:
        row = self.db.execute("SELECT value FROM settings WHERE key='dataset'").fetchone()
        if row is None:
            raise ValueError("Import an intervention JSON first with --input FILE or import FILE.")
        return row[0]

    def candidates(self) -> list[dict]:
        rows = self.db.execute("SELECT c.*, d.position FROM dataset_entries d JOIN candidates c ON c.id=d.candidate_id "
                               "WHERE d.dataset_id=? ORDER BY d.position", (self.active_dataset(),)).fetchall()
        return [self._candidate(row) for row in rows]

    def candidate(self, key: str) -> dict:
        row = self.db.execute("SELECT * FROM candidates WHERE id=?", (key,)).fetchone()
        if row is None:
            raise ValueError("Unknown intervention record.")
        return self._candidate(row)

    def resolve(self, value: str) -> dict:
        records = self.candidates()
        if value.isdigit():
            index = int(value)
            if 1 <= index <= len(records):
                return records[index-1]
        matches = [r for r in records if r["id"].startswith(value)]
        if len(matches) == 1:
            return matches[0]
        raise ValueError("Choose an intervention number from list, or a unique record-ID prefix.")

    def _candidate(self, row) -> dict:
        result = dict(row)
        version = self.db.execute("SELECT * FROM versions WHERE id=?", (result["current_version"],)).fetchone()
        result["data"] = json.loads(version["data_json"])
        result["origin"] = version["origin"]
        result["proposition"] = ("The supplied evidence supports retaining " + result["name"]
                                 + " as a research candidate for " + result["condition"] + ".")
        result["learned_knowledge"] = self.applied_knowledge(result)
        claim=result['data'].get('review_claim')
        if isinstance(claim,str) and claim.strip():
            result['proposition']=claim.strip()
        result['claim_approved']=any(f['field']=='review_claim' for f in result['learned_knowledge'])
        return result

    def current_reviews(self, candidate: dict) -> list[dict]:
        rows = [dict(r) for r in self.db.execute(
            "SELECT r.*,c.email AS clinician_email,c.provider AS identity_provider FROM reviews r "
            "JOIN clinicians c ON c.id=r.clinician_id WHERE r.candidate_id=? AND r.version_id=? "
            "AND r.id=(SELECT MAX(x.id) FROM reviews x WHERE x.candidate_id=r.candidate_id "
            "AND x.version_id=r.version_id AND x.reviewer_key=r.reviewer_key) "
            "AND r.decision!='withdrawn' ORDER BY r.id",
            (candidate["id"], candidate["current_version"]))]
        for row in rows:
            analysis = self.db.execute("SELECT result_json FROM feedback_analyses WHERE id=?", (row["analysis_id"],)).fetchone() if row["analysis_id"] else None
            row["reason_analysis"] = json.loads(analysis[0]) if analysis else None
        return rows

    def _save_assessment(self, candidate: dict, result: dict, trigger: str) -> dict:
        created = now()
        cursor = self.db.execute("INSERT INTO assessments(candidate_id,version_id,result_json,trigger,created_at) "
                                 "VALUES (?,?,?,?,?)", (candidate["id"], candidate["current_version"],
                                                       encoded(result), trigger, created))
        return dict(result, assessment_id=cursor.lastrowid, created_at=created, trigger=trigger)

    def latest_assessment(self, candidate: dict):
        row = self.db.execute("SELECT * FROM assessments WHERE candidate_id=? AND version_id=? ORDER BY id DESC LIMIT 1",
                              (candidate["id"], candidate["current_version"])).fetchone()
        if row is None:
            return None
        return dict(json.loads(row["result_json"]), assessment_id=row["id"], created_at=row["created_at"], trigger=row["trigger"])

    def review(self, candidate_id: str, version: str, reviewer: Clinician, decision: str,
               rationale: str, citation: str = "", *, analysis_id=None, analysis_confirmed=False, submitted_decision=None, decision_override=False, expected_source_digest=None, notes_only=False, concern_kind=None, concern_text=None) -> dict:
        self.require_clinician(reviewer)
        identity = reviewer.id
        if decision not in DECISIONS:
            raise ValueError("Choose support, oppose, uncertain or withdrawn.")
        if not rationale.strip() and not notes_only:
            raise ValueError("A short reason is required to make the review traceable.")
        if notes_only and (analysis_id or analysis_confirmed or decision_override):
            raise ValueError("A review note cannot apply a model interpretation or change source facts.")
        reason_mode = 'note' if notes_only else 'legacy'
        with self.transaction():
            candidate = self.candidate(candidate_id)
            if candidate["current_version"] != version:
                raise ValueError("The evidence changed while you were reviewing it. Use show and review the current version.")
            if analysis_confirmed and expected_source_digest is not None and self.evidence_digest(self.source_records(candidate))!=expected_source_digest:
                raise ValueError('Another reviewer changed the source facts. Inspect the new evidence before confirming.')
            if analysis_id:
                from app.services.feedback_omega import packet
                stored = self.db.execute("SELECT result_json FROM feedback_analyses WHERE id=?", (analysis_id,)).fetchone()
                if stored is None:
                    raise ValueError("Unknown Omega interpretation; no review saved.")
                result = json.loads(stored[0])
                from app.services.feedback_reason_schema import validate
                validate(result["analysis"], result["context"])
                expected = packet(candidate, submitted_decision, rationale.strip(), citation.strip(), clinician=reviewer)
                actual = dict(result["context"])
                actual.pop("sources", None)
                actual["prior_reviews"] = []
                if expected != actual:
                    raise ValueError("The Omega interpretation belongs to another review or evidence version.")
                allowed = result["analysis"]["proposed_decision"] if analysis_confirmed else submitted_decision
                if decision_override and (not analysis_confirmed or decision not in ("support", "oppose", "uncertain")):
                    raise ValueError("An explicit final judgment requires a confirmed reason interpretation.")
                if not decision_override and decision != allowed:
                    raise ValueError("Confirm the proposed assessment or keep the submitted decision explicitly.")
            elif analysis_confirmed or decision_override:
                raise ValueError("A confirmed analysis ID is required.")
            previous = self.db.execute("SELECT * FROM reviews WHERE candidate_id=? AND version_id=? AND reviewer_key=? "
                                       "ORDER BY id DESC LIMIT 1", (candidate_id, version, identity)).fetchone()
            if decision == "withdrawn" and (previous is None or previous["decision"] == "withdrawn"):
                raise ValueError("This reviewer has no current judgment to withdraw.")
            active_previous=previous if previous and previous['decision']!='withdrawn' else None
            concern_kind=(active_previous['concern_kind'] if active_previous else '') if concern_kind is None else concern_kind.strip()
            concern_text=(active_previous['concern_text'] if active_previous else '') if concern_text is None else concern_text.strip()
            if concern_kind not in ('',*CONCERNS):raise ValueError('Choose a supported record concern type.')
            if bool(concern_kind)!=bool(concern_text):raise ValueError('A record concern needs both its category and a description; clear both to resolve your concern.')
            before = self.belief_assessment(candidate)
            source_before = self.latest_evidence_score(candidate)
            if previous and (previous["decision"], previous["rationale"], previous["citation"], previous["analysis_id"], bool(previous["analysis_confirmed"]), bool(previous["decision_override"]), previous["reason_mode"], previous["concern_kind"], previous["concern_text"]) == (decision, rationale.strip(), citation.strip(), analysis_id, bool(analysis_confirmed), bool(decision_override), reason_mode, concern_kind, concern_text):
                self.db.execute("DELETE FROM review_drafts WHERE candidate_id=? AND version_id=? AND clinician_id=?", (candidate_id, version, reviewer.id))
                return {"before": source_before, "after": source_before, "belief_before":before,"belief_after":before,"changed": False}
            cursor = self.db.execute("INSERT INTO reviews(candidate_id,version_id,reviewer,reviewer_key,clinician_id,decision,rationale,"
                                     "citation,supersedes_id,created_at,analysis_id,analysis_confirmed,submitted_decision,decision_override,reason_mode,concern_kind,concern_text) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                     (candidate_id, version, reviewer.name, identity, reviewer.id, decision, rationale.strip(), citation.strip(),
                                      previous["id"] if previous else None, now(), analysis_id, int(analysis_confirmed), submitted_decision, int(decision_override), reason_mode, concern_kind, concern_text))
            # Opinions remain in reviews; no clinician becomes a numerical
            # study observation. Old agreement calculations remain in history.
            source_after=source_before
            evidence_changed=False
            if analysis_id and analysis_confirmed and result.get('analysis',{}).get('source_assessments') is not None:
                source_after,evidence_changed=self._apply_source_plan(candidate,reviewer,self.source_plan(candidate,result),analysis_id,
                    f"Source facts confirmed with review #{cursor.lastrowid}: {reviewer.name}")
            after = self._rebuild_belief(candidate, f"Review #{cursor.lastrowid}: {decision}; rebuild current appraisal premises")
            self.db.execute("DELETE FROM review_drafts WHERE candidate_id=? AND version_id=? AND clinician_id=?", (candidate_id, version, reviewer.id))
        return {"before":source_before,"after":source_after,"belief_before":before,"belief_after":after,
                "changed":True,"evidence_changed":evidence_changed}

    def history(self, candidate: dict) -> dict:
        key = candidate["id"]
        return {
            "belief_assessments": self.belief_history(candidate),
            "reviews": [dict(r) for r in self.db.execute("SELECT * FROM reviews WHERE candidate_id=? ORDER BY id", (key,))],
            "assessments": [dict(json.loads(r["result_json"]), assessment_id=r["id"], version_id=r["version_id"],
                                  created_at=r["created_at"], trigger=r["trigger"])
                            for r in self.db.execute("SELECT * FROM assessments WHERE candidate_id=? ORDER BY id", (key,))],
            "corrections": [dict(r) for r in self.db.execute("SELECT * FROM corrections WHERE candidate_id=? ORDER BY id", (key,))],
            "learned_knowledge": self.learning_details(candidate),
            "reason_interpretations": [dict(row, result=json.loads(row["result_json"])) for row in self.db.execute(
                "SELECT DISTINCT a.* FROM feedback_analyses a JOIN reviews r ON r.analysis_id=a.id WHERE r.candidate_id=? ORDER BY a.created_epoch", (key,))],
        }

    def propose_correction(self, candidate_id: str, version: str, reviewer: Clinician, field: str,
                           replacement: str, rationale: str, citation: str = "", *, reusable=False, grounding=None) -> int:
        self.require_clinician(reviewer)
        if field not in CORRECTION_FIELDS:
            raise ValueError("Correct one of: " + ", ".join(sorted(CORRECTION_FIELDS)))
        if not replacement.strip() or not rationale.strip():
            raise ValueError("A replacement and reason are required.")
        if reusable and (field not in REUSABLE_FIELDS or not public_reference(citation)):
            raise ValueError("Reusable evidence corrections need a supported field and a public PMID, NCT, DOI or HTTPS source reference.")
        with self.transaction():
            candidate = self.candidate(candidate_id)
            if candidate["current_version"] != version:
                raise ValueError("Evidence changed; propose the correction against the current version.")
            if field=='review_claim':
                if not reusable:raise ValueError('Claim changes use explicit shared approval. Use claim N.')
                grounding=self.validate_claim_grounding(candidate,grounding)
                if citation.strip()!=grounding['source_id']:raise ValueError('The claim citation must match its supporting quote.')
            if candidate["data"].get(field) is not None and not isinstance(candidate["data"][field], str):
                raise ValueError("This CLI only corrects text fields.")
            if reusable:
                if candidate["origin"] != "upstream" and not candidate["origin"].startswith(("learned knowledge", "knowledge removed")):
                    raise ValueError("This assessment has a patient-specific local correction. Use correct for local changes; teach must start from a source or learned evidence version.")
                if candidate["data"].get(field) == replacement.strip():
                    raise ValueError("The replacement is already present; no new knowledge was proposed.")
            cursor = self.db.execute("INSERT INTO corrections(candidate_id,base_version,field,replacement,rationale,citation,"
                                     "proposer,proposer_key,proposer_clinician_id,created_at,reusable,grounding_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                                     (candidate_id, version, field, replacement.strip(), rationale.strip(), citation.strip(),
                                      reviewer.name, reviewer.id, reviewer.id, now(), int(reusable),encoded(grounding) if grounding else None))
        return cursor.lastrowid

    def resolve_correction(self, correction_id: int, reviewer: Clinician, accept: bool, rationale: str) -> dict:
        self.require_clinician(reviewer)
        identity = reviewer.id
        if not rationale.strip():
            raise ValueError("Explain the correction decision.")
        knowledge_id = None
        with self.transaction():
            row = self.db.execute("SELECT * FROM corrections WHERE id=?", (correction_id,)).fetchone()
            if row is None or row["status"] != "pending":
                raise ValueError("Choose a pending correction ID from corrections.")
            if row["proposer_key"] == identity:
                raise ValueError("A different authenticated clinician must accept or reject this correction. Use login to change accounts.")
            candidate = self.candidate(row["candidate_id"])
            if row["base_version"] != candidate["current_version"]:
                raise ValueError("This correction targets older evidence. Submit a new proposal against the current version.")
            if accept and row['field']=='review_claim':
                self.validate_claim_grounding(candidate,json.loads(row['grounding_json'] or 'null'))
            before = self.latest_assessment(candidate)
            status = "accepted" if accept else "rejected"
            self.db.execute("UPDATE corrections SET status=?,resolver=?,resolver_clinician_id=?,resolution_reason=?,resolved_at=? WHERE id=?",
                            (status, reviewer.name, reviewer.id, rationale.strip(), now(), correction_id))
            if accept:
                if row["reusable"]:
                    knowledge_id = self._publish_knowledge(row, candidate, reviewer, rationale.strip())
                    self._refresh_known_candidates()
                else:
                    data = dict(candidate["data"])
                    data[row["field"]] = row["replacement"]
                    version = digest([candidate["id"], data, "correction", correction_id])
                    self.db.execute("INSERT INTO versions VALUES (?,?,?,?,?,?)",
                                    (version, candidate["id"], encoded(data), f"accepted correction #{correction_id}",
                                     candidate["current_version"], now()))
                    self.db.execute("UPDATE candidates SET current_version=? WHERE id=?", (version, candidate["id"]))
                candidate = self.candidate(candidate["id"])
                after = self.latest_assessment(candidate) if row["reusable"] else self._save_assessment(
                    candidate, assess([]), f"Correction #{correction_id} accepted; fresh review required")
            else:
                after = before
            self._rebuild_belief(candidate, f"Correction #{correction_id} resolved; current version")
        return {"candidate": candidate, "before": before, "after": after, "accepted": accept, "knowledge_id": knowledge_id}

    def pending_corrections(self) -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT k.*,c.name,c.condition,c.current_version FROM corrections k JOIN candidates c ON c.id=k.candidate_id "
            "JOIN dataset_entries d ON d.candidate_id=c.id WHERE k.status='pending' AND d.dataset_id=? ORDER BY k.id",
            (self.active_dataset(),))]

    def export(self, path: Path):
        path = path.expanduser().resolve()
        if path == self.path or path.exists():
            raise ValueError("Choose a new export filename; existing files are never overwritten.")
        dataset = self.db.execute("SELECT * FROM datasets WHERE id=?", (self.active_dataset(),)).fetchone()
        payload = {
            "schema": "alis-feedback-export-v1", "exported_at": now(),
            "research_support_only": True, "reviewer_identity": "Authenticated accounts; local test accounts are separate from ALIS clinician accounts",
            "original_handoff": json.loads(dataset["original_json"]),
            "file_review_access": [dict(row) for row in self.db.execute(
                "SELECT clinician_id,source_name,granted_at FROM file_review_access WHERE dataset_id=? ORDER BY granted_at,clinician_id",
                (dataset["id"],))],
            "records": [], "clinicians": [],
        }
        for candidate in self.candidates():
            history = self.history(candidate)
            versions = [dict(r) for r in self.db.execute("SELECT * FROM versions WHERE candidate_id=? ORDER BY created_at", (candidate["id"],))]
            payload["records"].append({"candidate": candidate, "evidence_assessment": self.latest_evidence_score(candidate),
                                       "source_evidence_events":self.evidence_events(candidate),
                                       "claim_appraisal": self.belief_assessment(candidate),
                                       "legacy_review_agreement": self.latest_assessment(candidate),
                                       "current_reviews": self.current_reviews(candidate), "versions": versions, **history})
        authors = {r["clinician_id"] for item in payload["records"] for r in item["reviews"] if r["clinician_id"]}
        authors.update(r[field] for item in payload["records"] for r in item["corrections"]
                       for field in ("proposer_clinician_id", "resolver_clinician_id") if r[field])
        authors.update(r["clinician_id"] for r in payload["file_review_access"])
        authors.update(r[field] for item in payload["records"] for r in item["learned_knowledge"]
                       for field in ("proposer_id", "approver_id"))
        authors.update(event["clinician_id"] for item in payload["records"] for r in item["learned_knowledge"] for event in r["events"])
        authors.update(event['clinician_id'] for item in payload['records'] for event in item['source_evidence_events'])
        payload["clinicians"] = [self.clinician_profile(identity) for identity in sorted(authors)]
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
