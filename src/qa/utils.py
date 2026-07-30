"""Small shared helpers: time formatting, resilient JSON parsing, retries."""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable

from google.genai import types

from qa.config import RETRY_ATTEMPTS, RETRY_BASE_DELAY_SEC, TEXT_MODEL

logger = logging.getLogger(__name__)


def convert_seconds_to_hhmmss(seconds: float) -> str:
    seconds = int(seconds)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def convert_hhmmss_to_seconds(hhmmss: str | float) -> float:
    """Parse ``HH:MM:SS`` (or ``MM:SS``) into seconds; numbers pass through.

    Fractional seconds are truncated, mirroring the Symphony helper.
    """
    if not isinstance(hhmmss, str):
        return float(hhmmss)
    hhmmss = hhmmss.split(".")[0]
    parts = hhmmss.split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid time format: {hhmmss!r}. Expected HH:MM:SS.")
    if len(parts) == 2:
        parts = ["00", *parts]
    hours, minutes, seconds = map(int, parts)
    return float(hours * 3600 + minutes * 60 + seconds)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        return text[7:].removesuffix("```").strip()
    if text.startswith("```"):
        return text[3:].removesuffix("```").strip()
    return text


def _get_json(json_string: str):
    """json.loads that also unwraps results that are themselves JSON strings."""
    result = json.loads(json_string)
    tries = 0
    while isinstance(result, str) and tries < 10:
        tries += 1
        result = json.loads(result)
    return result


JSON_REPAIR_PROMPT = (
    "The following string is a malformed JSON. "
    "Correct it and return only the valid JSON object:\n\n"
)


async def fix_and_parse_json(json_string: str | None, client) -> dict | None:
    """Parse JSON, asking Gemini to repair it on failure. Returns None if hopeless.

    Port of Symphony's fix_and_parse_json: strip markdown fences, parse, and on
    a decode error make one LLM repair call. Structured output makes this a
    backstop rather than the main path.
    """
    if not json_string:
        return None
    try:
        result = _get_json(strip_code_fences(json_string))
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        pass

    logger.warning("malformed JSON, attempting LLM repair: %r", json_string)
    try:
        response = await client.aio.models.generate_content(
            model=TEXT_MODEL,
            contents=JSON_REPAIR_PROMPT + json_string,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        if response.text is None:
            logger.error("JSON repair call returned no text")
            return None
        result = _get_json(strip_code_fences(response.text))
        return result if isinstance(result, dict) else None
    except Exception:
        logger.exception("failed to fix and parse JSON after LLM attempt")
        return None


async def with_retries[T](
    fn: Callable[[], Awaitable[T]], attempts: int = RETRY_ATTEMPTS
) -> T:
    """Run async ``fn()`` with exponential backoff on any exception.

    Replaces Symphony's 5×(60s, doubling) transport retry; per-request
    timeouts come from the injected Gemini client.
    """
    delay = RETRY_BASE_DELAY_SEC
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except Exception:
            if attempt == attempts:
                raise
            logger.warning(
                "attempt %d/%d failed, retrying in %.1fs",
                attempt,
                attempts,
                delay,
                exc_info=True,
            )
            await asyncio.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable: retry loop exited without returning")
