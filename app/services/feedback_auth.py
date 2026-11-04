"""Clinician authentication for the feedback CLI; credentials stay out of traces."""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
import getpass
import hashlib
import hmac
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid
import warnings


class AuthenticationError(RuntimeError):
    pass


class AuthenticationUnavailable(AuthenticationError):
    """Retrying credentials cannot fix an unavailable account service."""


class LoginRejected(AuthenticationError):
    """The server rejected authentication or access; never infer account existence."""


class PasswordInputError(AuthenticationError):
    """A terminal with hidden input is required; do not loop on credentials."""


class AccountAlreadyExists(AuthenticationError):
    pass


class AccountCreationUnconfirmed(AuthenticationError):
    """A server may have committed signup; never report a definite failure."""


class AccountCreatedLoginError(AuthenticationError):
    """Registration succeeded independently of the subsequent login."""


class InvalidServerResponse(AuthenticationError):
    pass


@dataclass(frozen=True)
class Clinician:
    id: str
    provider: str
    subject: str
    issuer: str
    name: str
    email: str
    clinic_id: str | None = None


@dataclass
class LoginSession:
    clinician: Clinician
    store: object = field(repr=False)
    token: str | None = field(default=None, repr=False)
    expires_at: float | None = None
    authorised_patient: str | None = None

    def require(self, patient_id: str | None = None):
        self.store.require_clinician(self.clinician)
        if self.expires_at is not None and self.expires_at <= time.time():
            raise AuthenticationError("Your ALIS session expired. Use login to authenticate again.")
        if self.clinician.provider == "alis" and patient_id:
            # The backend, not a locally decoded token, authorises this patient.
            response = request_json(self.clinician.issuer, "/patients/" + str(uuid.UUID(patient_id)), token=self.token)
            if not isinstance(response, dict) or str(response.get("id", response.get("patient_id", ""))) != patient_id:
                raise AuthenticationError("ALIS did not confirm access to this patient.")
            self.authorised_patient = patient_id
            self.store.reconnect()


def normalise_email(value: str) -> str:
    email = value.strip().casefold()
    if len(email) > 254 or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        raise AuthenticationError("Enter a valid email address.")
    return email


def password_hash(password: str) -> str:
    if len(password) < 10:
        raise AuthenticationError("Use at least 10 characters for a local-account password.")
    salt = secrets.token_bytes(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 600000)
    return "pbkdf2-sha256-v1$600000$" + salt.hex() + "$" + key.hex()


def password_matches(password: str, stored: str) -> bool:
    try:
        marker, rounds, salt, expected = stored.split("$")
        if marker != "pbkdf2-sha256-v1" or rounds != "600000":
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(rounds))
        return hmac.compare_digest(actual.hex(), expected)
    except (ValueError, TypeError):
        return False


def register_local(store, name: str, email: str, password: str) -> Clinician:
    email = normalise_email(email)
    if not name.strip() or len(name.strip()) > 120:
        raise AuthenticationError("Give a display name of 1–120 characters.")
    identity = str(uuid.uuid4())
    clinician = Clinician(identity, "local", identity, "local-feedback", name.strip(), email)
    store.register_local_clinician(clinician, password_hash(password))
    return clinician


def login_local(store, email: str, password: str) -> LoginSession:
    email = normalise_email(email)
    row = store.local_account(email)
    # Use the same work factor when no account exists, and never expose which
    # part of a credential pair was wrong.
    stored = row["password_hash"] if row else "pbkdf2-sha256-v1$600000$" + "00"*16 + "$" + "00"*32
    if not password_matches(password, stored) or row is None or not row["is_active"]:
        raise AuthenticationError("Invalid email or password, or inactive account.")
    clinician = Clinician(row["id"], "local", row["subject"], row["issuer"], row["name"], row["email"])
    store.record_login(clinician.id)
    return LoginSession(clinician, store)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def api_url(value: str) -> str:
    value = value.strip().rstrip("/")
    url = urlsplit(value)
    try:
        loopback = ipaddress.ip_address(url.hostname or "").is_loopback
    except ValueError:
        loopback = url.hostname == "localhost"
    if (url.scheme not in ("http", "https") or not url.hostname or url.username or url.password
            or url.query or url.fragment or (url.scheme == "http" and not loopback)):
        raise AuthenticationError("Use HTTPS for ALIS login, or HTTP on localhost/127.0.0.1.")
    return value


def request_json(base: str, path: str, payload=None, token=None):
    base = api_url(base)
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = "Bearer " + token
    request = Request(base + path, data=json.dumps(payload).encode() if payload is not None else None,
                      headers=headers, method="POST" if payload is not None else "GET")
    try:
        with build_opener(NoRedirect()).open(request, timeout=15) as response:
            result = json.loads(response.read(2*1024*1024))
    except HTTPError as error:
        # Never echo response bodies: validation errors may contain credentials.
        error.close()
        if error.code in (401, 403):
            raise LoginRejected("ALIS denied this login or access. Check the account and permissions.") from None
        if error.code == 409:
            raise AccountAlreadyExists("Account ALREADY EXISTS in ALIS. No additional account was created; log in with that email.") from None
        if error.code == 422:
            raise AuthenticationError("ALIS rejected the account fields. Check the email, name and password.") from None
        if error.code >= 500:
            raise AuthenticationUnavailable(f"ALIS is unavailable (HTTP {error.code}); login stopped. Try again when the backend is running.") from None
        raise AuthenticationError(f"ALIS request failed (HTTP {error.code}).") from None
    except (URLError, TimeoutError, OSError):
        raise AuthenticationUnavailable("Cannot reach ALIS. Start the backend or use --auth local for separate local test accounts.") from None
    except (ValueError, UnicodeError):
        raise InvalidServerResponse("ALIS returned an invalid response.") from None
    if not isinstance(result, dict):
        raise InvalidServerResponse("ALIS returned an invalid account response.")
    return result


def issued_claims(result: dict) -> tuple[str, dict]:
    """Read identity only from a token just issued by the trusted login endpoint.

    Arbitrary pasted tokens are not accepted. Patient access is subsequently
    validated by the backend with that same token. No JWT signing secret is
    copied into the chatbot or CLI.
    """
    token = result.get("access_token")
    try:
        part = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
        subject = str(uuid.UUID(claims["sub"]))
        clinic = str(uuid.UUID(claims["clinic_id"]))
        if (claims.get("actor") != "clinician" or claims.get("clinician_id") != subject
                or not isinstance(claims.get("name"), str) or not claims["name"].strip()
                or not isinstance(claims.get("email"), str)
                or not isinstance(claims.get("exp"), (int, float)) or claims["exp"] <= time.time()):
            raise ValueError
        return token, dict(claims, sub=subject, clinic_id=clinic)
    except (AttributeError, IndexError, KeyError, ValueError, TypeError):
        raise AuthenticationError("ALIS did not issue a valid clinician session. Clinic accounts cannot submit clinician reviews.") from None


def login_alis(store, base: str, email: str, password: str) -> LoginSession:
    base = api_url(base)
    email = normalise_email(email)
    token, claims = issued_claims(request_json(base, "/clinicians/login", {"email": email, "password": password}))
    if claims["email"].strip().casefold() != email:
        raise AuthenticationError("ALIS returned a different clinician identity; login was rejected.")
    identity = str(uuid.uuid5(uuid.NAMESPACE_URL, base + "/clinicians/" + claims["sub"]))
    clinician = Clinician(identity, "alis", claims["sub"], base, claims["name"], email, claims["clinic_id"])
    store.remember_alis_clinician(clinician)
    return LoginSession(clinician, store, token=token, expires_at=claims["exp"])


def register_alis(base: str, clinic_token: str, name: str, email: str, password: str) -> dict:
    """Use ALIS's existing clinician-registration endpoint and its returned ID."""
    email = normalise_email(email)
    try:
        created = request_json(base, "/clinicians", {"name": name, "email": email, "password": password}, token=clinic_token)
    except AccountAlreadyExists:
        raise
    except (AuthenticationUnavailable, InvalidServerResponse):
        raise AccountCreationUnconfirmed("Account creation could NOT BE CONFIRMED because ALIS became unavailable. Try logging in before repeating signup.") from None
    except AuthenticationError as error:
        raise AuthenticationError("Clinician signup was NOT completed. " + str(error)) from None
    try:
        identity = str(uuid.UUID(created["id"]))
        clinic = str(uuid.UUID(created["clinic_id"]))
        if normalise_email(created["email"]) != email:
            raise ValueError
    except (KeyError, ValueError, TypeError, AuthenticationError):
        raise AccountCreationUnconfirmed("ALIS accepted the request but returned no valid account confirmation. Try logging in before repeating signup.") from None
    return {"id": identity, "clinic_id": clinic, "name": name, "email": email, "provider": "alis"}


def configured_api(root: Path) -> str:
    if os.getenv("ALIS_API_URL"):
        return os.environ["ALIS_API_URL"]
    environment = root / ".env"
    if environment.exists():
        for line in environment.read_text().splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == "ALIS_API_URL":
                return value.strip().strip("\"'")
    return "http://127.0.0.1:5001"


def hidden_password(prompt: str) -> str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            return getpass.getpass(prompt)
    except getpass.GetPassWarning:
        raise PasswordInputError("Hidden password input is unavailable. Run this command in an interactive terminal.") from None


def clinic_token_from(result: dict) -> str:
    token = result.get("access_token")
    try:
        part = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(part + "="*(-len(part)%4)))
        uuid.UUID(claims["sub"])
        if claims.get("actor") != "clinic":
            raise ValueError
        return token
    except (AttributeError, IndexError, KeyError, ValueError, TypeError):
        raise AuthenticationError("ALIS did not issue a clinic administrator session.") from None


def clinic_login_interactively(base: str) -> str:
    email = normalise_email(input("Clinic email: "))
    password = hidden_password("Clinic password (hidden): ")
    try:
        try:
            result = request_json(base, "/clinics/login", {"email": email, "password": password})
        except LoginRejected:
            raise LoginRejected(
                "Clinic login was rejected. Use an existing clinic administrator's email and password. "
                "If you have not created a clinic, choose option 3 to create the clinic and its first clinician. "
                "If you already have a clinician account, choose option 1 to sign in."
            ) from None
        token = clinic_token_from(result)
    finally:
        del password
    print("Clinic login successful.")
    return token


def create_clinic_interactively(base: str) -> str:
    print("Create a clinic account in ALIS.")
    name = input("Clinic name: ").strip()
    email = normalise_email(input("Clinic email: "))
    if not name:
        raise AuthenticationError("Clinic name is required; no account was created.")
    for attempt in range(3):
        password = hidden_password("Choose password (hidden, at least 8 characters): ")
        try:
            if len(password) < 8:
                print("Use at least 8 characters. No account was created.")
                continue
            if password != hidden_password("Confirm password (hidden): "):
                print("Passwords did not match. No account was created.")
                continue
            try:
                request_json(base, "/clinics", {"name": name, "email": email, "password": password})
            except (AuthenticationUnavailable, InvalidServerResponse):
                raise AccountCreationUnconfirmed("Clinic creation could not be confirmed. Try clinic login before repeating signup.") from None
            print("Clinic account CREATED successfully: " + name + " <" + email + ">")
            try:
                # The signup token lacks the regular clinic role claims. Obtain
                # the normal login token, exactly as the earlier ALIS CLI did.
                return clinic_token_from(request_json(base, "/clinics/login", {"email": email, "password": password}))
            except AuthenticationError:
                raise AccountCreatedLoginError("Clinic WAS CREATED, but automatic login failed. Choose clinic login (option 2); do not register it again.") from None
        finally:
            del password
    raise AuthenticationError("Clinic signup stopped; no account was created.")


def create_account_interactively(store, provider: str, base: str, clinic_token=None) -> LoginSession:
    if provider == "alis" and clinic_token is None:
        print("Add a clinician to an existing ALIS clinic.")
        print("Enter the existing clinic administrator's credentials below, not the new clinician's details.")
        print("First time using ALIS? Choose option 3 in the account menu to create a clinic and its first clinician.")
        try:
            clinic_token = clinic_login_interactively(base)
        except AuthenticationUnavailable as error:
            raise AuthenticationUnavailable("No clinician account was created: clinic administrator login is unavailable. " + str(error)) from None
        except AuthenticationError as error:
            raise AuthenticationError("No clinician account was created: clinic administrator login did not complete. " + str(error)) from None
    name = input("Clinician name: ").strip()
    if not name or len(name) > 120:
        raise AuthenticationError("Give a clinician name of 1–120 characters. No account was created.")
    email = normalise_email(input("Clinician email: "))
    for attempt in range(3):
        minimum = 10 if provider == "local" else 8
        password = hidden_password(f"Choose password (hidden, at least {minimum} characters): ")
        try:
            if len(password) < minimum:
                print(f"Use at least {minimum} characters. No account has been created.")
                continue
            if password != hidden_password("Confirm password (hidden): "):
                print("Passwords did not match. Re-enter your new password.")
                continue
            if provider == "alis":
                created = register_alis(base, clinic_token, name, email, password)
            else:
                account = register_local(store, name, email, password)
                created = {"id": account.id, "name": account.name, "email": account.email, "provider": "local"}
            print("Account CREATED successfully: " + created["name"] + " <" + created["email"] + ">")
            print(("ALIS clinician ID: " if provider == "alis" else "Local account ID: ") + created["id"])
            try:
                session = login_alis(store, base, email, password) if provider == "alis" else login_local(store, email, password)
            except AuthenticationError:
                raise AccountCreatedLoginError("The account WAS CREATED, but automatic sign-in failed. Use login with the new email; do not create it again.") from None
            print("Signed in as " + session.clinician.name + ".")
            return session
        finally:
            del password
    raise AuthenticationError("Account creation stopped after three unsuccessful password entries. Restart to try again; no account was created.")


def interactive_login(store, provider: str, base: str, create_account: bool = False) -> LoginSession:
    store.bind_auth(provider, api_url(base) if provider == "alis" else "local-feedback")
    from app.services.feedback_terminal import Terminal
    ui=Terminal()
    ui.begin('Clinician login','ALIS · '+base if provider=='alis' else 'Local test accounts · '+store.path.name)
    ui.text('Passwords are hidden. Reviews stay associated with your account.',kind='muted')
    if provider == "local" and store.local_account_count() == 0:
        print("No local test account exists in this database yet.")
        print("First-time setup: create your account below. ALIS accounts are accessed using --auth alis.")
        return create_account_interactively(store, provider, base)
    if create_account:
        return create_account_interactively(store, provider, base)
    attempts = 0
    while attempts < 3:
        if provider == "alis":
            ui.menu([('1','Log in as a clinician'),('2','Log in as a clinic'),
                     ('3','First-time setup · clinic + first clinician'),('4','Add clinician · clinic admin login required'),('q','Quit')])
        else:
            ui.menu([('1','Log in'),('2','Create a local test account'),('q','Quit')])
        choice = input("Choose an option: ").strip().casefold()
        if choice == "q":
            raise EOFError
        try:
            if choice in ("1", "login"):
                email = input("Clinician email: ").strip()
                password = hidden_password("Password (hidden): ")
                try:
                    session = login_alis(store, base, email, password) if provider == "alis" else login_local(store, email, password)
                finally:
                    del password
                return session
            if provider == "alis" and choice == "2":
                clinic_token = clinic_login_interactively(base)
                print("Clinic administration session. Clinician accounts submit intervention reviews.")
                if input("Create a clinician for this clinic now? [y/N]: ").strip().casefold() == "y":
                    return create_account_interactively(store, provider, base, clinic_token=clinic_token)
                continue
            if provider == "alis" and choice == "3":
                clinic_token = create_clinic_interactively(base)
                print("Next, create your clinician account under this clinic to review interventions.")
                return create_account_interactively(store, provider, base, clinic_token=clinic_token)
            if choice in (("4", "signup", "register") if provider == "alis" else ("2", "signup", "register")):
                return create_account_interactively(store, provider, base)
            print("Choose one of the displayed options, or q.")
        except (AuthenticationUnavailable, PasswordInputError, AccountCreationUnconfirmed):
            raise
        except AuthenticationError as error:
            attempts += 1
            print(str(error))
            if attempts < 3:
                print(f"{3-attempts} attempt(s) remain. Choose the appropriate login or account-creation option above.")
    if provider == "alis":
        raise AuthenticationError("Login stopped after three unsuccessful attempts. Restart and choose option 3 for first-time setup, option 1 for an existing clinician, or option 4 with an existing clinic administrator's login.")
    raise AuthenticationError("Login stopped after three unsuccessful attempts. Check the account database and login provider, or use --register to create an account.")
