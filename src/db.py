"""Postgres connection pool (Supabase).

One asyncpg pool per process, created lazily so the API and the Celery workers
share the same accessor without either needing a startup hook. Celery tasks are
synchronous entry points that ``asyncio.run`` an async body, which gets its own
event loop per task — so the pool is keyed by the running loop and rebuilt if a
different one asks for it, rather than blowing up with "attached to a different
loop".
"""

import asyncio
import logging
import os

import asyncpg

logger = logging.getLogger(__name__)

# Kept small: the API runs several replicas and every Celery worker holds one
# too, and Supabase's pooler has a connection ceiling.
POOL_MIN_SIZE = 1
POOL_MAX_SIZE = 8

# Statements are simple and indexed; anything slower than this is a problem we
# want surfaced as an error rather than a hung request.
COMMAND_TIMEOUT_SEC = 30.0

_pool: asyncpg.Pool | None = None
_pool_loop: asyncio.AbstractEventLoop | None = None
_pool_lock: asyncio.Lock | None = None


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return url


async def get_pool() -> asyncpg.Pool:
    """The pool for the current event loop, created on first use."""
    global _pool, _pool_loop, _pool_lock

    loop = asyncio.get_running_loop()
    if _pool is not None and _pool_loop is loop:
        return _pool

    if _pool_lock is None or _pool_loop is not loop:
        _pool_lock = asyncio.Lock()
    async with _pool_lock:
        if _pool is not None and _pool_loop is loop:
            return _pool
        if _pool is not None:
            # A previous loop's pool (a finished Celery task); drop it rather
            # than reuse connections bound to a loop that no longer runs.
            logger.debug("db: discarding pool from a previous event loop")
            with_suppressed_close = _pool.terminate
            with_suppressed_close()
        logger.info("db: creating connection pool (max=%d)", POOL_MAX_SIZE)
        _pool = await asyncpg.create_pool(
            database_url(),
            min_size=POOL_MIN_SIZE,
            max_size=POOL_MAX_SIZE,
            command_timeout=COMMAND_TIMEOUT_SEC,
        )
        _pool_loop = loop
    return _pool


async def close_pool() -> None:
    """Close the pool, if any. Called from the API's lifespan shutdown."""
    global _pool, _pool_loop
    if _pool is not None:
        await _pool.close()
        _pool = None
        _pool_loop = None
        logger.info("db: connection pool closed")
