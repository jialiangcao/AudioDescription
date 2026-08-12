"""Supabase JWT verification.

The backend never mints tokens: the browser signs in with supabase-js and sends
the resulting access token, and every ``/api`` route resolves it to a user id
that scopes each query. Supabase issues two flavours depending on project age —
asymmetric keys published at a JWKS endpoint, and the legacy shared HS256
secret — so both are supported and the configuration picks one.

There is also a development escape hatch (``ADESC_DEV_USER_ID``) that accepts
every request as one fixed user. It exists so the pipeline can be worked on
without standing up Supabase; it is off unless explicitly set, and it announces
itself loudly at startup because enabling it in a deployed environment would
hand the whole API to anyone.
"""

import logging
import os

import jwt
from fastapi import Header, HTTPException, Query
from jwt import PyJWKClient

logger = logging.getLogger(__name__)

# Supabase stamps this on every access token it issues.
EXPECTED_AUDIENCE = "authenticated"

_jwk_client: PyJWKClient | None = None


def dev_user_id() -> str | None:
    return os.environ.get("ADESC_DEV_USER_ID") or None


def _jwt_secret() -> str | None:
    return os.environ.get("SUPABASE_JWT_SECRET") or None


def _jwks_url() -> str | None:
    explicit = os.environ.get("SUPABASE_JWKS_URL")
    if explicit:
        return explicit
    base = os.environ.get("SUPABASE_URL")
    return f"{base.rstrip('/')}/auth/v1/.well-known/jwks.json" if base else None


def _get_jwk_client() -> PyJWKClient:
    """Cached JWKS client — it caches signing keys, so build it only once."""
    global _jwk_client
    if _jwk_client is None:
        url = _jwks_url()
        if not url:
            raise RuntimeError("neither SUPABASE_JWT_SECRET nor SUPABASE_URL is set")
        _jwk_client = PyJWKClient(url, cache_keys=True)
        logger.info("auth: verifying tokens against %s", url)
    return _jwk_client


def verify_token(token: str) -> str:
    """Return the user id (``sub``) a token belongs to, or raise 401."""
    dev_user = dev_user_id()
    if dev_user:
        return dev_user

    try:
        secret = _jwt_secret()
        if secret:
            claims = jwt.decode(
                token, secret, algorithms=["HS256"], audience=EXPECTED_AUDIENCE
            )
        else:
            signing_key = _get_jwk_client().get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["ES256", "RS256"],
                audience=EXPECTED_AUDIENCE,
            )
    except jwt.PyJWTError as exc:
        # Deliberately vague to the caller; the reason is for our logs only.
        logger.info("auth: rejected token: %s", exc)
        raise HTTPException(status_code=401, detail="invalid or expired token") from exc

    subject = claims.get("sub")
    if not subject:
        raise HTTPException(status_code=401, detail="token has no subject")
    return subject


async def current_user(authorization: str = Header(default="")) -> str:
    """FastAPI dependency: the authenticated user id for this request."""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        if dev_user_id():
            return verify_token("")
        raise HTTPException(status_code=401, detail="missing bearer token")
    return verify_token(token)


async def websocket_user(token: str = Query(default="")) -> str:
    """Same, for a websocket.

    Browsers cannot set headers on a WebSocket, so the token arrives as a query
    parameter. The client gets a short-lived, single-use ticket for this from
    ``POST /api/jobs/{id}/ws-ticket`` rather than putting its session JWT in a
    URL, where it would land in access logs and browser history.
    """
    if not token:
        if dev_user_id():
            return verify_token("")
        raise HTTPException(status_code=401, detail="missing token")
    return verify_token(token)


def warn_if_insecure() -> None:
    """Called once at startup, so a misconfiguration is impossible to miss."""
    if dev_user_id():
        logger.warning(
            "AUTH IS DISABLED: ADESC_DEV_USER_ID=%s is set, so every request is "
            "accepted as that user. Never set this outside local development.",
            dev_user_id(),
        )
    elif not _jwt_secret() and not _jwks_url():
        logger.warning(
            "auth: no SUPABASE_JWT_SECRET or SUPABASE_URL configured; every "
            "authenticated request will be rejected"
        )
