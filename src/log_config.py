"""Central logging configuration for the adesc backend.

Call ``configure_logging()`` once at process start (the server does this in its
lifespan). Every module only ever does ``logger = logging.getLogger(__name__)``
and logs against it — modules never touch handlers or levels themselves, so
tests and embedders keep full control of output.

The level is read from the ``ADESC_LOG_LEVEL`` env var (default ``INFO``); set it
to ``DEBUG`` to see the verbose per-shot / per-frame / per-segment traces, or to
``WARNING`` to see only problems. Third-party libraries that log floods at INFO
(httpx request lines, faster-whisper's model chatter) are pinned to WARNING so
our own progress logs stay readable.
"""

import logging
import os

DEFAULT_LEVEL = "INFO"

# Chatty third-party loggers held one notch quieter than our own default so the
# pipeline's stage/progress logs aren't buried. Overridden if the caller asks
# for DEBUG explicitly (see below).
_NOISY_LOGGERS = ("httpx", "httpcore", "faster_whisper", "urllib3", "PIL")

_configured = False


def configure_logging(level: str | None = None, *, force: bool = False) -> None:
    """Install a level + formatter on the root logger (idempotent).

    ``level`` overrides the ``ADESC_LOG_LEVEL`` env var; both default to
    ``INFO``. Safe to call more than once — subsequent calls are no-ops unless
    ``force=True``, which reconfigures handlers (useful in tests).
    """
    global _configured
    if _configured and not force:
        return

    resolved = (level or os.getenv("ADESC_LOG_LEVEL") or DEFAULT_LEVEL).upper()
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=force,
    )

    # Keep third-party floods below our level unless we're explicitly debugging.
    if resolved != "DEBUG":
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True
