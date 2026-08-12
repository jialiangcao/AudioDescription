"""Global Gemini rate limiting and a typed retry policy.

Two problems the single-process version never had to solve.

**Rate limiting.** Concurrency used to be capped by two independent
``asyncio.Semaphore``s inside one process, which is meaningless once several
worker machines call Gemini at once — the effective rate becomes machines ×
per-process concurrency, and the quota is hit by whoever gets there first. The
token bucket here lives in Redis, so every worker draws from one shared
allowance.

**Retries.** ``qa.utils.with_retries`` caught *every* exception and backed off
identically, so a 400 (a prompt Gemini will never accept) burned the same three
attempts and six seconds as a 503. Retries are now typed: transient statuses
back off with jitter and honour ``Retry-After``; everything else fails
immediately.

Jitter matters here specifically. Stage 2 fans out per shot across workers, so a
quota trip tends to hit many calls at once; without jitter they would all wake
together and trip it again.
"""

import asyncio
import logging
import os
import random
import time
from collections.abc import Awaitable, Callable

import events

logger = logging.getLogger(__name__)

# Requests per minute allowed across the whole deployment. Sized to the Gemini
# tier in use; the point is that it is one number, not one per process.
DEFAULT_RPM = int(os.environ.get("GEMINI_RPM", "1000"))

RETRY_ATTEMPTS = 4
RETRY_BASE_DELAY_SEC = 1.0
RETRY_MAX_DELAY_SEC = 60.0

# Statuses worth trying again: rate limiting, and the transient server-side
# failures Gemini returns under load. Anything else (400 malformed request, 401
# bad key, 403 no access, 404 unknown model) will fail identically on a retry.
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

# How long to wait for a token before giving up, so a wedged bucket surfaces as
# a stage failure rather than a hung worker.
ACQUIRE_TIMEOUT_SEC = 300

# Refill and take in one round trip. Redis runs this atomically, so concurrent
# workers cannot both see the last token.
#
#   KEYS[1] bucket   ARGV[1] capacity   ARGV[2] refill/sec
#   ARGV[3] now (s)  ARGV[4] tokens wanted
# Returns the seconds to wait, or 0 when the tokens were granted.
_TAKE_SCRIPT = """
local bucket = KEYS[1]
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local want = tonumber(ARGV[4])

local state = redis.call('HMGET', bucket, 'tokens', 'updated')
local tokens = tonumber(state[1])
local updated = tonumber(state[2])
if tokens == nil then
  tokens = capacity
  updated = now
end

tokens = math.min(capacity, tokens + (now - updated) * rate)
local wait = 0
if tokens >= want then
  tokens = tokens - want
else
  wait = (want - tokens) / rate
end

redis.call('HMSET', bucket, 'tokens', tokens, 'updated', now)
-- Expire an idle bucket; a fresh one starts full, which is correct because
-- nothing has been spent for at least this long.
redis.call('EXPIRE', bucket, 120)
return tostring(wait)
"""

_script = None


def _bucket_key() -> str:
    return "gemini:rpm"


async def _take(tokens: int = 1) -> float:
    """Try to take ``tokens``; returns the seconds to wait, 0 if granted."""
    global _script
    redis = events.get_redis()
    if _script is None:
        _script = redis.register_script(_TAKE_SCRIPT)
    wait = await _script(
        keys=[_bucket_key()],
        args=[DEFAULT_RPM, DEFAULT_RPM / 60.0, time.time(), tokens],
    )
    return float(wait)


async def acquire(tokens: int = 1, timeout_sec: float = ACQUIRE_TIMEOUT_SEC) -> None:
    """Block until the shared bucket allows ``tokens``.

    Falls open if Redis is unreachable: throttling is a cost control, and
    failing every job because the limiter is down would be the worse outcome.
    """
    deadline = time.monotonic() + timeout_sec
    while True:
        try:
            wait = await _take(tokens)
        except Exception:
            logger.warning(
                "gemini limiter unavailable, proceeding unthrottled", exc_info=True
            )
            return
        if wait <= 0:
            return
        if time.monotonic() + wait > deadline:
            raise TimeoutError(f"waited {timeout_sec}s for Gemini rate-limit capacity")
        # Jitter so workers that queued together don't wake together.
        await asyncio.sleep(wait * random.uniform(1.0, 1.25))


# --------------------------------------------------------------------------- #
# retries
# --------------------------------------------------------------------------- #


def _status_of(exc: Exception) -> int | None:
    """The HTTP status behind a google-genai error, if there is one."""
    for attribute in ("code", "status_code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _retry_after_of(exc: Exception) -> float | None:
    """A server-specified wait, which is always better than our guess."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        raw = headers.get("Retry-After") or headers.get("retry-after")
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def is_retryable(exc: Exception) -> bool:
    """Whether trying this exact call again could plausibly succeed."""
    if isinstance(exc, TimeoutError | ConnectionError | asyncio.TimeoutError):
        return True
    status = _status_of(exc)
    if status is not None:
        return status in RETRYABLE_STATUS
    # No status at all usually means the request never completed (a transport
    # or DNS failure), which is worth one more attempt.
    return isinstance(exc, OSError)


async def with_retries[T](
    fn: Callable[[], Awaitable[T]],
    attempts: int = RETRY_ATTEMPTS,
    tokens: int = 1,
) -> T:
    """Rate-limit and run ``fn()``, retrying only what is worth retrying.

    Replaces the previous catch-all backoff, which spent the same three
    attempts on a malformed request as on a 503.
    """
    delay = RETRY_BASE_DELAY_SEC
    for attempt in range(1, attempts + 1):
        await acquire(tokens)
        try:
            return await fn()
        except Exception as exc:
            if attempt == attempts or not is_retryable(exc):
                if not is_retryable(exc):
                    logger.warning("gemini call failed unretryably: %r", exc)
                raise
            wait = _retry_after_of(exc) or delay * random.uniform(1.0, 1.5)
            wait = min(wait, RETRY_MAX_DELAY_SEC)
            logger.warning(
                "gemini attempt %d/%d failed (%s), retrying in %.1fs",
                attempt,
                attempts,
                _status_of(exc) or type(exc).__name__,
                wait,
            )
            await asyncio.sleep(wait)
            delay = min(delay * 2, RETRY_MAX_DELAY_SEC)
    raise RuntimeError("unreachable: retry loop exited without returning")
