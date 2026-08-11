"""Short-lived, single-use tickets for authenticating a websocket.

A browser cannot set an Authorization header on a WebSocket, so the token has to
travel in the URL — and URLs end up in access logs, proxy logs and browser
history. Rather than put the user's session JWT there, the client exchanges it
over normal HTTP for a ticket that is random, expires in a minute, and is
consumed on first use.

Tickets live in Redis so any API replica can redeem one, and the redemption is
an atomic GETDEL: a replayed ticket finds nothing.
"""

import logging
import secrets

from events import get_redis

logger = logging.getLogger(__name__)

TICKET_TTL_SEC = 60
_PREFIX = "ws-ticket:"


async def issue(user_id: str, job_id: str) -> str:
    """Mint a ticket binding one user to one job."""
    ticket = secrets.token_urlsafe(32)
    await get_redis().setex(_PREFIX + ticket, TICKET_TTL_SEC, f"{user_id}:{job_id}")
    return ticket


async def redeem(ticket: str) -> tuple[str, str] | None:
    """Consume a ticket, returning ``(user_id, job_id)`` or None.

    GETDEL makes this single-use: a second attempt with the same ticket — a
    replay, or a reconnect that reused it — gets nothing back.
    """
    if not ticket:
        return None
    raw = await get_redis().getdel(_PREFIX + ticket)
    if raw is None:
        return None
    # The client is built with decode_responses=True, but the stubs still
    # allow bytes, so normalize rather than assume.
    text = raw.decode() if isinstance(raw, bytes) else str(raw)
    user_id, _, job_id = text.partition(":")
    return user_id, job_id
