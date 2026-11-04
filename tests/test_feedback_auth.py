"""Authentication boundary checks against a local ALIS-protocol HTTP fixture."""
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
import io
from unittest.mock import patch
import uuid

from app.services.feedback_auth import AccountAlreadyExists, AccountCreatedLoginError, AuthenticationError, AuthenticationUnavailable, api_url, interactive_login, login_alis, login_local, register_alis, register_local
from app.services.feedback_store import FeedbackStore

CLINICIAN = "d61995bc-0f24-4bfc-bbbc-28ac82852aa3"
CLINIC = "619479bd-8c7c-4ac4-99df-982ac8629156"
PATIENT = "a68bb8cf-aeea-4f65-a14d-09366ea0cd81"


def token_for(email, subject=CLINICIAN, name="Protocol fixture clinician"):
    claims = {"sub": subject, "clinician_id": subject, "clinic_id": CLINIC,
              "actor": "clinic" if email.startswith("clinic@") else "clinician",
              "name": name, "email": email, "exp": time.time()+3600}
    encode = lambda value: base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")
    return encode({"alg": "HS256"}) + "." + encode(claims) + ".fixture-signature"


class ALISProtocolFixture(BaseHTTPRequestHandler):
    accounts = {}
    def log_message(self, *args):
        pass

    def respond(self, status, value):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(value).encode())

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/clinics/login" and payload.get("password") == "Fixture-password-only":
            self.respond(200, {"access_token": token_for("clinic@example.test")})
            return
        if self.path == "/clinicians":
            if not self.headers.get("Authorization", "").startswith("Bearer "):
                self.respond(403, {"detail": "Clinic login required"})
                return
            email = payload["email"]
            if email in self.accounts:
                self.respond(409, {"detail": "Email already exists"})
                return
            account = {"id": str(uuid.uuid5(uuid.NAMESPACE_DNS, email)), "clinic_id": CLINIC,
                       "name": payload["name"], "email": email, "password": payload["password"]}
            self.accounts[email] = account
            self.respond(201, {k: v for k, v in account.items() if k != "password"})
            return
        account = self.accounts.get(payload.get("email"))
        expected = account["password"] if account else "Fixture-password-only"
        if (self.path != "/clinicians/login" or payload.get("password") != expected
                or payload.get("email", "").startswith("cannot-sign-in@")):
            self.respond(401, {"detail": "private-password private-token"})
            return
        self.respond(200, {"access_token": token_for(payload["email"], account["id"], account["name"]) if account else token_for(payload["email"]), "token_type": "bearer"})

    def do_GET(self):
        if self.path != "/patients/" + PATIENT or not self.headers.get("Authorization", "").startswith("Bearer "):
            self.respond(403, {"detail": "private-token"})
            return
        self.respond(200, {"id": PATIENT})


class FeedbackAuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), ALISProtocolFixture)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = "http://127.0.0.1:" + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        ALISProtocolFixture.accounts.clear()
        self.directory = tempfile.TemporaryDirectory()
        self.store = FeedbackStore(Path(self.directory.name)/"feedback.sqlite3")

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def test_alis_login_identity_is_stable_and_token_is_not_persisted(self):
        first = login_alis(self.store, self.url, "clinician@example.test", "Fixture-password-only")
        first.require(PATIENT)
        second = login_alis(self.store, self.url, "CLINICIAN@example.test", "Fixture-password-only")
        self.assertEqual(first.clinician.id, second.clinician.id)
        self.assertEqual(first.clinician.subject, CLINICIAN)
        self.assertEqual(first.clinician.clinic_id, CLINIC)
        self.assertNotIn(first.token, repr(first))
        row = self.store.db.execute("SELECT * FROM clinicians").fetchone()
        self.assertIsNone(row["password_hash"])
        self.assertNotIn("access_token", row.keys())

    def test_backend_patient_access_is_enforced(self):
        session = login_alis(self.store, self.url, "clinician@example.test", "Fixture-password-only")
        with self.assertRaises(AuthenticationError):
            session.require(str(uuid.uuid4()))

    def test_clinic_actor_cannot_become_review_author(self):
        with self.assertRaisesRegex(AuthenticationError, "Clinic accounts cannot"):
            login_alis(self.store, self.url, "clinic@example.test", "Fixture-password-only")
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM clinicians").fetchone()[0], 0)

    def test_wrong_password_error_does_not_echo_response_body(self):
        with self.assertRaises(AuthenticationError) as error:
            login_alis(self.store, self.url, "clinician@example.test", "Wrong-password")
        self.assertNotIn("private-", str(error.exception))

    def test_rejected_clinic_signup_explains_first_time_setup_without_creating_account(self):
        output = io.StringIO()
        with patch("builtins.input", side_effect=["4", "unregistered@example.test", "q"]), \
             patch("app.services.feedback_auth.hidden_password", return_value="Wrong-password"), \
             redirect_stdout(output), self.assertRaises(EOFError):
            interactive_login(self.store, "alis", self.url)
        message = output.getvalue()
        self.assertIn("existing clinic administrator's credentials", message)
        self.assertIn("choose option 3 to create the clinic and its first clinician", message)
        self.assertIn("choose option 1 to sign in", message)
        self.assertNotIn("Wrong-password", message)
        self.assertNotIn("private-", message)
        self.assertNotIn("Account CREATED", message)
        self.assertEqual(ALISProtocolFixture.accounts, {})

    def test_remote_plain_http_is_rejected(self):
        with self.assertRaises(AuthenticationError):
            api_url("http://example.com")
        with self.assertRaises(AuthenticationError):
            api_url("https://user:password@example.com")

    def test_local_accounts_cannot_be_added_to_alis_database(self):
        login_alis(self.store, self.url, "clinician@example.test", "Fixture-password-only")
        with self.assertRaisesRegex(AuthenticationError, "different login provider"):
            register_local(self.store, "Local name", "local@example.test", "Local-test-password")

    def test_local_login_preserves_account_identity_and_hides_password(self):
        clinician = register_local(self.store, "Clinician name", "local@example.test", "Local-test-password")
        session = login_local(self.store, "LOCAL@example.test", "Local-test-password")
        self.assertEqual(session.clinician.id, clinician.id)
        session.require()
        stored = self.store.local_account("local@example.test")["password_hash"]
        self.assertTrue(stored.startswith("pbkdf2-sha256-v1$600000$"))
        self.assertNotIn("Local-test-password", stored)
        with self.assertRaises(AuthenticationError):
            login_local(self.store, "local@example.test", "Incorrect-password")

    def test_existing_alis_signup_creates_multiple_distinct_clinicians(self):
        ids = []
        for name, email in (("Clinician One", "one@example.test"), ("Clinician Two", "two@example.test")):
            receipt = register_alis(self.url, token_for("clinic@example.test"), name, email, "New-account-password")
            session = login_alis(self.store, self.url, email, "New-account-password")
            self.assertEqual(receipt["id"], session.clinician.subject)
            ids.append(receipt["id"])
        self.assertEqual(len(set(ids)), 2)
        self.assertEqual(len(ALISProtocolFixture.accounts), 2)

    def test_duplicate_signup_reports_existing_account(self):
        token = token_for("clinic@example.test")
        register_alis(self.url, token, "One", "one@example.test", "New-account-password")
        with self.assertRaisesRegex(AccountAlreadyExists, "ALREADY EXISTS"):
            register_alis(self.url, token, "One", "one@example.test", "New-account-password")
        self.assertEqual(len(ALISProtocolFixture.accounts), 1)

    def test_creation_confirmation_is_distinct_from_automatic_login(self):
        output = io.StringIO()
        with patch("builtins.input", side_effect=["clinic@example.test", "New clinician", "cannot-sign-in@example.test"]), \
             patch("app.services.feedback_auth.hidden_password", side_effect=["Fixture-password-only", "New-account-password", "New-account-password"]), \
             redirect_stdout(output), self.assertRaisesRegex(AccountCreatedLoginError, "WAS CREATED"):
            interactive_login(self.store, "alis", self.url, create_account=True)
        self.assertIn("Account CREATED successfully", output.getvalue())
        self.assertIn("ALIS clinician ID:", output.getvalue())
        self.assertEqual(len(ALISProtocolFixture.accounts), 1)
        self.assertNotIn("New-account-password", output.getvalue())


class FeedbackLoginFlowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = FeedbackStore(Path(self.directory.name)/"feedback.sqlite3")

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def test_empty_database_goes_directly_to_account_creation(self):
        output = io.StringIO()
        with patch("builtins.input", side_effect=["First user", "first@example.test"]), \
             patch("app.services.feedback_auth.hidden_password", side_effect=["Synthetic-password", "Synthetic-password"]), \
             redirect_stdout(output):
            session = interactive_login(self.store, "local", "")
        self.assertEqual(session.clinician.email, "first@example.test")
        self.assertEqual(self.store.local_account_count(), 1)
        self.assertIn("No local test account exists", output.getvalue())
        self.assertNotIn("1. Log in", output.getvalue())
        self.assertNotIn("Synthetic-password", output.getvalue())

    def test_invalid_password_does_not_loop_indefinitely(self):
        register_local(self.store, "User", "user@example.test", "Synthetic-password")
        with patch("builtins.input", side_effect=["1", "user@example.test"]*3), \
             patch("app.services.feedback_auth.hidden_password", return_value="Incorrect-password"), \
             redirect_stdout(io.StringIO()), self.assertRaisesRegex(AuthenticationError, "stopped after three"):
            interactive_login(self.store, "local", "")
        self.assertEqual(self.store.local_account_count(), 1)

    def test_offline_backend_stops_instead_of_repeating_menu(self):
        with patch("builtins.input", side_effect=["1", "user@example.test"]), \
             patch("app.services.feedback_auth.hidden_password", return_value="Synthetic-password"), \
             patch("app.services.feedback_auth.login_alis", side_effect=AuthenticationUnavailable("Backend offline")) as login, \
             redirect_stdout(io.StringIO()), self.assertRaises(AuthenticationUnavailable):
            interactive_login(self.store, "alis", "http://127.0.0.1:5001")
        login.assert_called_once()


if __name__ == "__main__":
    unittest.main()
