import time
import uuid

import jwt
import pytest
from fastapi import HTTPException

import auth

# 32+ bytes: PyJWT warns below the RFC 7518 minimum for HS256.
SECRET = "test-signing-secret-long-enough-for-hs256"


def _token(secret=SECRET, **overrides):
    claims = {
        "sub": str(uuid.uuid4()),
        "aud": auth.EXPECTED_AUDIENCE,
        "exp": int(time.time()) + 3600,
        **overrides,
    }
    return jwt.encode(claims, secret, algorithm="HS256"), claims


@pytest.fixture(autouse=True)
def _hs256(monkeypatch):
    monkeypatch.delenv("ADESC_DEV_USER_ID", raising=False)
    monkeypatch.setenv("SUPABASE_JWT_SECRET", SECRET)
    auth._jwk_client = None
    yield
    auth._jwk_client = None


def test_a_valid_token_resolves_to_its_subject():
    token, claims = _token()
    assert auth.verify_token(token) == claims["sub"]


def test_a_token_signed_with_the_wrong_key_is_rejected():
    token, _ = _token(secret="a-different-secret-also-long-enough-here")

    with pytest.raises(HTTPException) as excinfo:
        auth.verify_token(token)
    assert excinfo.value.status_code == 401


def test_an_expired_token_is_rejected():
    token, _ = _token(exp=int(time.time()) - 60)

    with pytest.raises(HTTPException) as excinfo:
        auth.verify_token(token)
    assert excinfo.value.status_code == 401


def test_a_token_for_another_audience_is_rejected():
    """Supabase stamps aud=authenticated; anything else isn't a user session."""
    token, _ = _token(aud="some-other-service")

    with pytest.raises(HTTPException):
        auth.verify_token(token)


def test_a_token_without_a_subject_is_rejected():
    claims = {
        "aud": auth.EXPECTED_AUDIENCE,
        "exp": int(time.time()) + 3600,
    }
    token = jwt.encode(claims, SECRET, algorithm="HS256")

    with pytest.raises(HTTPException) as excinfo:
        auth.verify_token(token)
    assert excinfo.value.status_code == 401


def test_rejection_does_not_explain_why():
    """The reason goes to the logs, not to whoever is probing."""
    token, _ = _token(secret="another-wrong-secret-of-sufficient-length")

    with pytest.raises(HTTPException) as excinfo:
        auth.verify_token(token)
    assert excinfo.value.detail == "invalid or expired token"


async def test_current_user_requires_a_bearer_scheme():
    token, _ = _token()

    with pytest.raises(HTTPException):
        await auth.current_user(authorization=token)  # no "Bearer " prefix
    with pytest.raises(HTTPException):
        await auth.current_user(authorization="")

    assert await auth.current_user(authorization=f"Bearer {token}")


async def test_websocket_user_reads_the_token_from_the_query():
    token, claims = _token()

    assert await auth.websocket_user(token=token) == claims["sub"]
    with pytest.raises(HTTPException):
        await auth.websocket_user(token="")


# --------------------------------------------------------------------------- #
# the development escape hatch
# --------------------------------------------------------------------------- #


async def test_dev_user_id_accepts_every_request(monkeypatch):
    dev_user = str(uuid.uuid4())
    monkeypatch.setenv("ADESC_DEV_USER_ID", dev_user)

    assert auth.verify_token("obviously-not-a-jwt") == dev_user
    assert await auth.current_user(authorization="") == dev_user
    assert await auth.websocket_user(token="") == dev_user


def test_dev_mode_announces_itself(monkeypatch, caplog):
    """Enabling this in a deployed environment would hand over the whole API."""
    monkeypatch.setenv("ADESC_DEV_USER_ID", str(uuid.uuid4()))

    with caplog.at_level("WARNING"):
        auth.warn_if_insecure()

    assert "AUTH IS DISABLED" in caplog.text


def test_missing_configuration_warns_rather_than_failing_silently(monkeypatch, caplog):
    monkeypatch.delenv("ADESC_DEV_USER_ID", raising=False)
    monkeypatch.delenv("SUPABASE_JWT_SECRET", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_JWKS_URL", raising=False)

    with caplog.at_level("WARNING"):
        auth.warn_if_insecure()

    assert "no SUPABASE_JWT_SECRET" in caplog.text
