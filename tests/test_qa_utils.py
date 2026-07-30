import pytest
from conftest import FakeGeminiResponse

from qa import utils
from qa.utils import (
    convert_hhmmss_to_seconds,
    convert_seconds_to_hhmmss,
    fix_and_parse_json,
    strip_code_fences,
    with_retries,
)


def test_convert_seconds_to_hhmmss():
    assert convert_seconds_to_hhmmss(0) == "00:00:00"
    assert convert_seconds_to_hhmmss(5.9) == "00:00:05"
    assert convert_seconds_to_hhmmss(3661) == "01:01:01"


def test_convert_hhmmss_to_seconds():
    assert convert_hhmmss_to_seconds("00:03:21") == 201.0
    assert convert_hhmmss_to_seconds("03:21") == 201.0  # MM:SS tolerated
    assert convert_hhmmss_to_seconds("00:00:05.04") == 5.0  # fraction truncated
    assert convert_hhmmss_to_seconds(7.5) == 7.5  # numbers pass through
    with pytest.raises(ValueError):
        convert_hhmmss_to_seconds("42")


def test_strip_code_fences():
    assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fences('```\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fences('{"a": 1}') == '{"a": 1}'


async def test_fix_and_parse_json_direct(fake_gemini_client):
    assert await fix_and_parse_json('{"a": 1}', fake_gemini_client) == {"a": 1}
    assert await fix_and_parse_json('```json\n{"a": 1}\n```', fake_gemini_client) == {
        "a": 1
    }
    # Double-encoded JSON strings are unwrapped.
    assert await fix_and_parse_json('"{\\"a\\": 1}"', fake_gemini_client) == {"a": 1}
    assert await fix_and_parse_json(None, fake_gemini_client) is None
    assert fake_gemini_client.calls == []  # no LLM involved


async def test_fix_and_parse_json_repairs_via_llm(fake_gemini_client):
    fake_gemini_client.queue(FakeGeminiResponse('{"agent": "finish"}'))
    result = await fix_and_parse_json("{'agent': 'finish'", fake_gemini_client)
    assert result == {"agent": "finish"}
    assert len(fake_gemini_client.calls) == 1


async def test_fix_and_parse_json_gives_up(fake_gemini_client):
    fake_gemini_client.queue(FakeGeminiResponse("still not json"))
    assert await fix_and_parse_json("not json", fake_gemini_client) is None


async def test_with_retries_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(utils, "RETRY_BASE_DELAY_SEC", 0.0)
    attempts = []

    async def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("boom")
        return "ok"

    assert await with_retries(flaky) == "ok"
    assert len(attempts) == 3


async def test_with_retries_raises_after_exhaustion(monkeypatch):
    monkeypatch.setattr(utils, "RETRY_BASE_DELAY_SEC", 0.0)

    async def always_fails():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await with_retries(always_fails, attempts=2)
