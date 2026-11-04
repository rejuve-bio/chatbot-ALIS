"""Resume the installed local ALIS API without rebuilding it or its database."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import urlsplit

from app.services.feedback_auth import AuthenticationError, AuthenticationUnavailable, api_url, request_json


def ready(base: str) -> bool:
    response = request_json(base, "/health")
    if response.get("status") != "ok":
        raise AuthenticationUnavailable("ALIS has not reported a healthy API.")
    return True


def command(arguments, timeout=8):
    try:
        return subprocess.run(arguments, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        raise AuthenticationUnavailable("Could not run the local ALIS startup check. Start Docker Desktop and retry; no credentials were collected.") from None


def ensure_alis_ready(base: str, auto_start=True):
    base = api_url(base)
    print("Checking ALIS signup/login service…", flush=True)
    try:
        ready(base)
        print("ALIS signup/login service is ready.", flush=True)
        return
    except AuthenticationError:
        parsed = urlsplit(base)
        installed_local = (parsed.hostname in ("127.0.0.1", "localhost", "::1")
                           and parsed.port == 5001 and parsed.path in ("", "/"))
        if not auto_start or not installed_local:
            raise AuthenticationUnavailable("ALIS is not ready at " + base + ". No credentials were collected. Start that backend and retry.") from None
    docker = shutil.which("docker")
    if not docker:
        raise AuthenticationUnavailable("ALIS is offline and Docker is not installed. No credentials were collected.")
    if command([docker, "info", "--format", "{{.ServerVersion}}"], timeout=5).returncode:
        application = Path("/Applications/Docker.app")
        if sys.platform != "darwin" or not application.exists():
            raise AuthenticationUnavailable("ALIS is offline. Start Docker and retry; no credentials were collected.")
        print("Starting Docker Desktop for your existing ALIS backend…", flush=True)
        if command(["open", "-a", str(application)]).returncode:
            raise AuthenticationUnavailable("Docker Desktop could not start. Open it and retry; no credentials were collected.")
        deadline = time.monotonic() + 90
        next_update = time.monotonic() + 10
        while command([docker, "info", "--format", "{{.ServerVersion}}"], timeout=5).returncode:
            if time.monotonic() >= deadline:
                raise AuthenticationUnavailable("Docker did not become ready. Finish Docker Desktop setup and retry; no credentials were collected.")
            if time.monotonic() >= next_update:
                print("Waiting for Docker Desktop…", flush=True)
                next_update = time.monotonic() + 10
            time.sleep(1)
    # Only resume these existing containers. Never invoke compose up, rebuild an
    # older checkout, initialise a database or replace stored login credentials.
    for container in ("alis-backend-db-1", "alis-backend-api-1"):
        state = command([docker, "container", "inspect", "--format", "{{.State.Running}}", container])
        if state.returncode:
            raise AuthenticationUnavailable("The installed ALIS container " + container + " is missing. Restore the existing backend before login; no containers or accounts were recreated.")
        if state.stdout.strip() != "true":
            print("Resuming " + container + "…", flush=True)
            if command([docker, "start", container], timeout=15).returncode:
                raise AuthenticationUnavailable("Could not resume " + container + ". No credentials were collected.")
    print("Waiting for the existing ALIS API…", flush=True)
    deadline = time.monotonic() + 60
    next_update = time.monotonic() + 10
    while True:
        try:
            ready(base)
            print("ALIS signup/login service is ready.", flush=True)
            return
        except AuthenticationError:
            if time.monotonic() >= deadline:
                raise AuthenticationUnavailable("The existing ALIS API did not become healthy. Check docker logs alis-backend-api-1; no credentials were collected.") from None
        if time.monotonic() >= next_update:
            print("Waiting for the existing ALIS API…", flush=True)
            next_update = time.monotonic() + 10
        time.sleep(1)
