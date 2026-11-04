"""Review lifecycle checks, using real native PLN for numerical updates."""
import json
from pathlib import Path
import runpy
import shutil
import tempfile
import unittest
from unittest.mock import patch

from app.services.feedback_pln import NativePLNError
from app.services.feedback_store import FeedbackStore
from app.services.feedback_auth import AuthenticationError, login_local, register_local

FeedbackSession = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/feedback"))["Session"]


def handoff():
    return {
        "patient_id": "synthetic-patient",
        "risk_areas": [
            {"disease_name": "Condition A", "investigational_evidence": [
                {"name": "Candidate X", "why": "Synthetic assertion, not medical evidence.",
                 "publications": [], "evidence": "unassessed"}]},
            {"disease_name": "Condition B", "treatments": [
                {"name": "Candidate X", "why": "Another context.", "publications": []}]},
        ],
    }


@unittest.skipUnless(shutil.which("swipl"), "Native SWI-Prolog is required for these lifecycle checks")
class ReviewLifecycle(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.input = self.root / "handoff.json"
        self.data = handoff()
        self.input.write_text(json.dumps(self.data))
        self.path = self.root / "reviews.sqlite3"
        self.store = FeedbackStore(self.path)
        self.store.import_file(self.input)
        self.clinicians = {}
        for name in ("Alice", "Bob", "Carol"):
            register_local(self.store, name, name.lower() + "@example.test", "Synthetic-password-2026")
            self.clinicians[name.lower()] = login_local(self.store, name.lower() + "@example.test", "Synthetic-password-2026").clinician

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def review(self, who, decision, target="1", reason="Synthetic workflow judgment"):
        candidate = self.store.resolve(target)
        return self.store.review(candidate["id"], candidate["current_version"], self.clinicians[who.strip().casefold()], decision, reason)

    def test_judgments_update_appraisal_ranking_without_becoming_source_effects(self):
        for who,verdict in [('Alice','support'),('Bob','oppose'),('Carol','uncertain')]:
            update=self.review(who,verdict)
            self.assertIsNone(update['after'])
        self.assertEqual(len(self.store.current_reviews(self.store.resolve('1'))),3)
        rows=self.store.rankings()[0]['ranked']
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['assessment']['strength'],.5)
        self.assertAlmostEqual(rows[0]['assessment']['confidence'],2/3)
        self.assertIsNone(self.store.latest_evidence_score(self.store.resolve('1')))
        self.assertFalse(self.review('bob','oppose')['changed'])

    def test_withdrawal_preserves_history_without_creating_a_scientific_score(self):
        self.review('Alice','support')
        result=self.review('Alice','withdrawn')['after']
        self.assertIsNone(result)
        self.assertEqual(len(self.store.history(self.store.resolve('1'))['reviews']),2)

    def test_same_candidate_in_different_context_is_separate(self):
        self.review("Alice", "support")
        self.assertIsNone(self.store.latest_assessment(self.store.resolve("2"))["strength"])
        self.data["patient_id"] = "another-synthetic-patient"
        self.input.write_text(json.dumps(self.data))
        self.store.import_file(self.input)
        self.assertIsNone(self.store.latest_assessment(self.store.resolve("1"))["strength"])

    def test_changed_source_invalidates_reviews_and_old_open_form(self):
        candidate = self.store.resolve("1")
        self.review("Alice", "support")
        self.data["risk_areas"][0]["investigational_evidence"][0]["why"] = "Changed assertion"
        self.input.write_text(json.dumps(self.data))
        imported = self.store.import_file(self.input)
        self.assertEqual(imported["changed"], 1)
        self.assertIsNone(self.store.latest_assessment(self.store.resolve("1"))["strength"])
        self.assertEqual(len(self.store.history(self.store.resolve("1"))["reviews"]), 1)
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            self.store.review(candidate["id"], candidate["current_version"], self.clinicians["bob"], "support", "Old form")

    def test_manual_opinion_does_not_require_or_fabricate_native_evidence(self):
        with patch('app.services.feedback_store.assess',side_effect=NativePLNError('Runtime unavailable')):
            self.assertIsNone(self.review('Alice','support')['after'])
        history=self.store.history(self.store.resolve('1'))
        self.assertEqual(len(history['reviews']),1)
        self.assertEqual(len(history['assessments']),1)

    def test_correction_preserves_source_and_reimport_does_not_erase_it(self):
        candidate = self.store.resolve("1")
        self.review("Alice", "support")
        correction = self.store.propose_correction(candidate["id"], candidate["current_version"], self.clinicians["alice"],
                                                  "why", "Corrected synthetic statement", "Correction reason")
        with self.assertRaisesRegex(ValueError, "different authenticated clinician"):
            self.store.resolve_correction(correction, self.clinicians["alice"], True, "Self approval")
        updated = self.store.resolve_correction(correction, self.clinicians["bob"], True, "Checked correction")
        self.assertIsNone(updated["after"]["strength"])
        self.store.import_file(self.input)
        self.assertEqual(self.store.resolve("1")["data"]["why"], "Corrected synthetic statement")
        self.assertEqual(json.loads(self.input.read_text()), handoff())
        self.assertEqual(len(self.store.history(self.store.resolve("1"))["reviews"]), 1)

    def test_new_session_and_export_preserve_reviews_and_native_trace(self):
        self.review("Alice", "support")
        self.store.close()
        self.store = FeedbackStore(self.path)
        candidate = self.store.resolve("1")
        self.assertEqual(self.store.current_reviews(candidate)[0]["reviewer"], "Alice")
        path = self.root / "export.json"
        self.store.export(path)
        export = json.loads(path.read_text())
        self.assertEqual(export["original_handoff"], handoff())
        self.assertEqual(export["records"][0]["current_reviews"][0]["clinician_id"], self.clinicians["alice"].id)
        self.assertNotIn("password_hash", path.read_text())
        self.assertNotIn("Synthetic-password-2026", path.read_text())
        self.assertIsNone(export["records"][0]["evidence_assessment"])
        self.assertIn("legacy_review_agreement",export["records"][0])
        with self.assertRaises(ValueError):
            self.store.export(path)

    def test_failed_import_does_not_replace_active_handoff(self):
        duplicate = self.data["risk_areas"][0]["investigational_evidence"][0]
        self.data["risk_areas"][0]["investigational_evidence"].append(duplicate)
        self.input.write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.store.import_file(self.input)
        self.assertEqual(len(self.store.candidates()), 2)

    def test_explicit_file_access_is_per_clinician_and_survives_restart(self):
        alice = self.clinicians["alice"]
        bob = self.clinicians["bob"]
        with self.assertRaisesRegex(AuthenticationError, "Open the supplied JSON"):
            self.store.require_file_review(alice)
        self.store.import_file(self.input, file_reviewer=alice)
        self.store.require_file_review(alice)
        self.review("Alice", "support")
        with self.assertRaises(AuthenticationError):
            self.store.require_file_review(bob)
        self.store.import_file(self.input, file_reviewer=bob)
        self.store.require_file_review(bob)
        self.assertEqual(self.store.current_reviews(self.store.resolve("1"))[0]["clinician_id"], alice.id)
        self.store.close()
        self.store = FeedbackStore(self.path)
        self.store.require_file_review(alice)
        self.store.require_file_review(bob)
        self.data["risk_areas"][0]["investigational_evidence"][0]["why"] = "New file version"
        self.input.write_text(json.dumps(self.data))
        self.store.import_file(self.input, file_reviewer=alice)
        with self.assertRaises(AuthenticationError):
            self.store.require_file_review(bob)

    def test_file_session_import_checks_identity_without_live_patient_lookup(self):
        authentication = login_local(self.store, "alice@example.test", "Synthetic-password-2026")
        session = FeedbackSession(self.store, authentication, "local", "", access_mode="file")
        with patch.object(authentication, "require", wraps=authentication.require) as require:
            result = session.import_handoff(self.input)
            session.require_access()
        self.assertEqual(result["total"], 2)
        self.assertTrue(all(not call.args for call in require.call_args_list))
        session.authentication = None
        with self.assertRaises(AuthenticationError):
            session.import_handoff(self.input)
        with self.assertRaises(AuthenticationError):
            session.require_access()

    def test_patient_session_still_enforces_backend_patient_permission(self):
        authentication = login_local(self.store, "alice@example.test", "Synthetic-password-2026")
        session = FeedbackSession(self.store, authentication, "alis", "http://127.0.0.1:5001", access_mode="patient")
        with patch.object(authentication, "require", side_effect=[None, AuthenticationError("Patient denied")]) as require:
            with self.assertRaisesRegex(AuthenticationError, "Patient denied"):
                session.import_handoff(self.input)
        self.assertEqual(require.call_args_list[-1].args, (self.data["patient_id"],))
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM file_review_access").fetchone()[0], 0)

    def test_correction_outside_active_file_is_not_resolvable(self):
        self.store.import_file(self.input, file_reviewer=self.clinicians["alice"])
        candidate = self.store.resolve("1")
        self.data["patient_id"] = "another-synthetic-patient"
        self.input.write_text(json.dumps(self.data))
        self.store.import_file(self.input, file_reviewer=self.clinicians["alice"])
        with self.assertRaisesRegex(AuthenticationError, "outside the opened review file"):
            self.store.require_active_candidate(candidate["id"])

    def test_name_alone_is_not_an_authenticated_review(self):
        candidate = self.store.resolve("1")
        with self.assertRaises(AuthenticationError):
            self.store.review(candidate["id"], candidate["current_version"], "Alice", "support", "A name is not a login")
        with self.assertRaises(AuthenticationError):
            login_local(self.store, "alice@example.test", "Wrong-password")
        self.assertEqual(self.store.history(candidate)["reviews"], [])

    def test_same_display_name_does_not_merge_distinct_accounts(self):
        register_local(self.store, "Alice", "another@example.test", "Another-synthetic-password")
        other = login_local(self.store, "another@example.test", "Another-synthetic-password").clinician
        self.review("Alice", "support")
        candidate = self.store.resolve("1")
        result = self.store.review(candidate["id"], candidate["current_version"], other, "oppose", "Different authenticated account")
        self.assertIsNone(result["after"])
        self.assertEqual({r['decision'] for r in self.store.current_reviews(candidate)},{'support','oppose'})
        current = self.store.current_reviews(candidate)
        self.assertEqual(len({r["clinician_id"] for r in current}), 2)
        self.assertEqual({r["reviewer"] for r in current}, {"Alice"})

    def test_legacy_reviews_are_preserved_but_not_promoted_to_authenticated(self):
        self.review("Alice", "support")
        self.store.db.execute("UPDATE reviews SET clinician_id=NULL")
        self.store.db.execute("PRAGMA user_version=1")
        self.store.close()
        self.store = FeedbackStore(self.path)
        candidate = self.store.resolve("1")
        self.assertEqual(self.store.current_reviews(candidate), [])
        self.assertIsNone(self.store.latest_assessment(candidate)["strength"])
        self.assertEqual(len(self.store.history(candidate)["reviews"]), 1)

    def teach(self, replacement="Synthetic corrected source interpretation"):
        candidate = self.store.resolve("1")
        correction = self.store.propose_correction(
            candidate["id"], candidate["current_version"], self.clinicians["alice"], "why", replacement,
            "Synthetic source interpretation correction", "PMID:12345", reusable=True)
        return self.store.resolve_correction(correction, self.clinicians["bob"], True, "Synthetic independent approval")

    def test_approved_knowledge_reused_across_handoffs_but_patient_votes_are_not(self):
        self.review("Alice", "support")
        learned = self.teach()
        self.assertEqual(learned["knowledge_id"], 1)
        self.review("Alice", "support")
        self.data["patient_id"] = "new-synthetic-patient"
        self.input.write_text(json.dumps(self.data))
        imported = self.store.import_file(self.input, file_reviewer=self.clinicians["bob"])
        self.assertEqual(imported["learned"], 1)
        candidate = self.store.resolve("1")
        self.assertEqual(candidate["data"]["why"], "Synthetic corrected source interpretation")
        self.assertIsNone(self.store.latest_assessment(candidate)["strength"])
        self.assertEqual(self.store.current_reviews(candidate), [])
        self.assertEqual(json.loads(self.input.read_text())["risk_areas"][0]["investigational_evidence"][0]["why"],
                         "Synthetic assertion, not medical evidence.")
        self.store.close()
        self.store = FeedbackStore(self.path)
        trace = self.store.learning_details(self.store.resolve("1"))
        self.assertTrue(trace[0]["applied"])
        self.assertEqual((trace[0]["proposer"], trace[0]["approver"]), ("Alice", "Bob"))
        self.assertNotIn("correction_id", trace[0])

    def test_reusable_knowledge_requires_source_and_another_clinician(self):
        candidate = self.store.resolve("1")
        for field, citation in (("why", ""), ("warning", "PMID:12345")):
            with self.assertRaisesRegex(ValueError, "public PMID"):
                self.store.propose_correction(candidate["id"], candidate["current_version"], self.clinicians["alice"],
                                               field, "Replacement", "Reason", citation, reusable=True)
        correction = self.store.propose_correction(candidate["id"], candidate["current_version"], self.clinicians["alice"],
                                                   "why", "Replacement", "Reason", "PMID:12345", reusable=True)
        with self.assertRaisesRegex(ValueError, "different authenticated clinician"):
            self.store.resolve_correction(correction, self.clinicians["alice"], True, "Self approval")
        self.store.resolve_correction(correction, self.clinicians["bob"], False, "Insufficient source support")
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM learned_knowledge").fetchone()[0], 0)

    def test_changed_source_or_claim_prevents_automatic_knowledge_reuse(self):
        self.teach()
        for change in ("source", "condition", "drugs", "intervention"):
            data = handoff()
            data["patient_id"] = "new-patient-" + change
            item = data["risk_areas"][0]["investigational_evidence"][0]
            if change == "source":
                item["publications"] = [{"pmid": "54321"}]
            elif change == "condition":
                data["risk_areas"][0]["disease_name"] = "Another condition"
            elif change == "drugs":
                item["drugs"] = ["Another compound"]
            else:
                item["name"] = "Another intervention"
            result = self.store.import_data(data, change + ".json")
            self.assertEqual(result["learned"], 0)
            self.assertEqual(self.store.resolve("1")["data"]["why"], "Synthetic assertion, not medical evidence.")

    def test_ordinary_patient_correction_never_becomes_shared_knowledge(self):
        candidate = self.store.resolve("1")
        correction = self.store.propose_correction(candidate["id"], candidate["current_version"], self.clinicians["alice"],
                                                   "why", "Patient-specific observation", "Local clinical reason")
        self.store.resolve_correction(correction, self.clinicians["bob"], True, "Checked for this patient")
        data = handoff()
        data["patient_id"] = "another-patient"
        result = self.store.import_data(data, "next.json")
        self.assertEqual(result["learned"], 0)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM learned_knowledge").fetchone()[0], 0)

    def test_knowledge_supersession_and_retirement_are_versioned(self):
        self.teach("First approved statement")
        first_version = self.store.resolve("1")["current_version"]
        self.teach("Updated approved statement")
        second_version = self.store.resolve("1")["current_version"]
        self.assertNotEqual(first_version, second_version)
        self.assertEqual(self.store.learning_details(self.store.resolve("1"))[0]["status"], "superseded")
        with self.assertRaisesRegex(ValueError, "who approved"):
            self.store.retire_knowledge(2, self.clinicians["alice"], "Proposer cannot revoke approval")
        count = self.store.retire_knowledge(2, self.clinicians["bob"], "Source interpretation withdrawn")
        self.assertEqual(count, 1)
        candidate = self.store.resolve("1")
        self.assertEqual(candidate["data"]["why"], "Synthetic assertion, not medical evidence.")
        self.assertNotEqual(candidate["current_version"], second_version)
        self.assertEqual(self.store.learning_details(candidate)[1]["events"][-1]["action"], "retired")
        self.assertIsNone(self.store.latest_assessment(candidate)["strength"])

    def test_local_override_survives_knowledge_retirement(self):
        self.teach()
        candidate = self.store.resolve("1")
        correction = self.store.propose_correction(candidate["id"], candidate["current_version"], self.clinicians["alice"],
                                                   "why", "Explicit local assessment", "Patient-specific reason")
        self.store.resolve_correction(correction, self.clinicians["bob"], True, "Accepted locally")
        self.store.retire_knowledge(1, self.clinicians["bob"], "Stop automatic reuse")
        self.assertEqual(self.store.resolve("1")["data"]["why"], "Explicit local assessment")


if __name__ == "__main__":
    unittest.main()
