"""Thin async wrappers over the injected Gemini client, plus ToolContext.

Mirrors the calling conventions of vision_analysis.py (client.aio, structured
output via response_schema, `response.text is None` guard). Every call runs at
temperature 0, matching Symphony.
"""

import asyncio
import logging
from dataclasses import dataclass

from google.genai import types

from qa.config import (
    TEXT_MAX_OUTPUT_TOKENS,
    TEXT_MODEL,
    THINKING_AGENT,
    THINKING_VISION,
    VISION_MAX_OUTPUT_TOKENS,
    VISION_MODEL,
)
from qa.frame_index import FrameIndex

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolContext:
    """Job-scoped state injected into tools by the agents.

    Symphony injected `frame_path`/`video_duration` by inspecting each tool's
    code object for parameter names; passing an explicit context keeps the
    same property (the model never supplies these) without the introspection.

    `blobs` resolves the frame index's blob keys to readable local files,
    pulling them from object storage on first use.
    """

    client: object
    frame_index: FrameIndex
    blobs: object


async def generate_text(
    client,
    *,
    system: str,
    user: str,
    schema=None,
    thinking=THINKING_AGENT,
    max_output_tokens: int = TEXT_MAX_OUTPUT_TOKENS,
) -> str:
    """One text-only Gemini call; returns the response text."""
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0.0,
        max_output_tokens=max_output_tokens,
        thinking_config=types.ThinkingConfig(thinking_level=thinking),
        response_mime_type="application/json" if schema is not None else None,
        response_schema=schema,
    )
    response = await client.aio.models.generate_content(
        model=TEXT_MODEL, contents=user, config=config
    )
    if response.text is None:
        raise RuntimeError(f"{TEXT_MODEL} returned no text")
    return response.text


async def generate_vision(
    client,
    *,
    system: str,
    user: str,
    frame_keys: list[str],
    blobs,
    schema=None,
    max_output_tokens: int = VISION_MAX_OUTPUT_TOKENS,
) -> str:
    """One Gemini call over a batch of frames + a prompt; returns the text.

    Frames arrive as blob keys and are resolved through ``blobs`` (a fetch is a
    no-op once the file is in scratch, which it is for every frame this job has
    already touched).
    """
    logger.debug("generate_vision: %d frame(s) -> %s", len(frame_keys), VISION_MODEL)

    def _read_all() -> list[bytes]:
        # One thread hop for the whole batch rather than one per frame: these
        # are cache hits in the common case, and hopping per frame would let
        # batches of different sizes interleave for no benefit.
        return [blobs.fetch(key).read_bytes() for key in frame_keys]

    contents: list = [
        types.Part.from_bytes(data=image, mime_type="image/jpeg")
        for image in await asyncio.to_thread(_read_all)
    ]
    contents.append(user)

    response = await client.aio.models.generate_content(
        model=VISION_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=THINKING_VISION),
            response_mime_type="application/json" if schema is not None else None,
            response_schema=schema,
        ),
    )
    if response.text is None:
        raise RuntimeError(f"{VISION_MODEL} returned no text")
    return response.text


async def generate_with_tools(
    client,
    *,
    system: str,
    contents: list,
    tools: list[types.Tool],
    thinking=THINKING_AGENT,
):
    """One Gemini call with function declarations bound; returns the raw
    response (callers read `.text` / `.function_calls` / `.candidates`).

    No python callables are passed, so the SDK's automatic function calling
    never triggers — the agents run the manual loop, like Symphony.
    """
    return await client.aio.models.generate_content(
        model=TEXT_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            max_output_tokens=TEXT_MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_level=thinking),
            tools=tools,
        ),
    )


def user_content(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


def tool_response_content(parts: list[types.Part]) -> types.Content:
    return types.Content(role="tool", parts=parts)
