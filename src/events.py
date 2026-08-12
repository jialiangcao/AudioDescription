"""Redis pub/sub fan-out for live job progress.

The old JobStore kept a per-job list of ``asyncio.Queue`` objects, so a client
only saw progress if its websocket happened to land on the process running the
job. Progress now goes two places: a durable row in ``job_events`` (which is
what a reconnecting client replays from) and a Redis publish on this channel
(which is what makes it *live*, from whichever worker produced it to whichever
API replica holds the socket).

Durability lives in Postgres, not here — a dropped publish costs a client some
latency, never an event, because the socket resumes from its last seq.
"""

import json
import logging
import os

import redis.asyncio as redis

logger = logging.getLogger(__name__)

_client: redis.Redis | None = None

# PING a pooled connection before reusing it if it has been idle this long.
#
# Managed Redis (and the Fly proxy in front of it) drops idle TCP connections,
# and redis-py does not notice a dead pooled connection until it tries to use
# it — so the next command after a quiet spell fails with a reset. This client
# is idle between jobs, which for this deployment is most of the time.
#
# The failure it prevents is a quiet one: publish() swallows every exception,
# and gemini_limits.acquire() falls open, so flapping connections would show up
# as missing live progress and unthrottled Gemini calls rather than an error.
HEALTH_CHECK_INTERVAL_SEC = 30


def redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def get_redis() -> redis.Redis:
    """The shared async Redis client, created on first use."""
    global _client
    if _client is None:
        _client = redis.from_url(
            redis_url(),
            decode_responses=True,
            health_check_interval=HEALTH_CHECK_INTERVAL_SEC,
        )
        logger.info("events: redis client ready")
    return _client


async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def channel(job_id: str) -> str:
    return f"job:{job_id}:events"


async def publish(job_id: str, event: dict) -> None:
    """Fan an event out to every API replica currently streaming this job.

    Never raises: the event is already committed to ``job_events``, so a Redis
    hiccup must not fail the pipeline stage that produced it. Subscribers will
    pick it up on their next reconnect-and-replay.
    """
    try:
        await get_redis().publish(channel(job_id), json.dumps(event))
    except Exception:
        logger.warning("events: publish failed for job %s", job_id, exc_info=True)
