"""Postgres connection pool (Supabase).

Pools are registered per event loop. asyncpg connections are bound to the loop
that created them, and this process has more than one caller: the API serves
requests on uvicorn's loop, while a Celery worker runs its async stage bodies on
a private loop of its own (see ``tasks._run``). A single global pool would be
torn down and rebuilt every time the other one asked for it.

Keeping one pool per loop means each is created once and reused for the life of
the process, which is what keeps a worker from opening fresh connections on
every task it picks up.
"""

import asyncio
import logging
import os
import weakref

import asyncpg

logger = logging.getLogger(__name__)

# Kept small: the API runs several replicas and every Celery worker holds one
# too, and Supabase's pooler has a connection ceiling.
POOL_MIN_SIZE = 1
POOL_MAX_SIZE = 8

# Statements are simple and indexed; anything slower than this is a problem we
# want surfaced as an error rather than a hung request.
COMMAND_TIMEOUT_SEC = 30.0

# loop -> pool. Weak-keyed so a finished loop's entry disappears with it.
_pools: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_locks: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return url


async def get_pool() -> asyncpg.Pool:
    """The pool for the current event loop, created on first use."""
    loop = asyncio.get_running_loop()
    pool = _pools.get(loop)
    if pool is not None:
        return pool

    lock = _locks.get(loop)
    if lock is None:
        lock = _locks.setdefault(loop, asyncio.Lock())
    async with lock:
        pool = _pools.get(loop)
        if pool is not None:
            return pool
        logger.info("db: creating connection pool (max=%d)", POOL_MAX_SIZE)
        pool = await asyncpg.create_pool(
            database_url(),
            min_size=POOL_MIN_SIZE,
            max_size=POOL_MAX_SIZE,
            command_timeout=COMMAND_TIMEOUT_SEC,
        )
        _pools[loop] = pool
    return pool


async def close_pool() -> None:
    """Close this loop's pool, if it has one."""
    loop = asyncio.get_running_loop()
    pool = _pools.pop(loop, None)
    if pool is not None:
        await pool.close()
        logger.info("db: connection pool closed")
