"""The shared Gemini rate limiter and typed retry policy."""

import pytest

import gemini_limits


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Backoff waits are asserted on, not slept through."""
    slept = []

    async def _sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(gemini_limits.asyncio, "sleep", _sleep)
    return slept


@pytest.fixture
def unlimited(monkeypatch):
    """Stub out the bucket so retry tests exercise only the retry loop.

    Deliberately not autouse: the bucket tests below call the real acquire().
    """

    async def _acquire(tokens=1, timeout_sec=None):
        return None

    monkeypatch.setattr(gemini_limits, "acquire", _acquire)


class ApiError(Exception):
    """Shaped like a google-genai error: a status code, optional response."""

    def __init__(self, code, retry_after=None):
        super().__init__(f"api error {code}")
        self.code = code
        if retry_after is not None:
            self.response = type(
                "Response", (), {"headers": {"Retry-After": retry_after}}
            )()


# --------------------------------------------------------------------------- #
# what is worth retrying
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504])
def test_transient_statuses_are_retryable(status):
    assert gemini_limits.is_retryable(ApiError(status))


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retryable(status):
    """A malformed request or a bad key fails identically on a retry.

    The old catch-all policy spent three attempts and six seconds on these.
    """
    assert not gemini_limits.is_retryable(ApiError(status))


def test_transport_failures_are_retryable():
    assert gemini_limits.is_retryable(ConnectionError("connection reset"))
    assert gemini_limits.is_retryable(TimeoutError())
    assert gemini_limits.is_retryable(OSError("dns failure"))


def test_a_plain_error_with_no_status_is_not_retryable():
    assert not gemini_limits.is_retryable(ValueError("bad argument"))


# --------------------------------------------------------------------------- #
# the retry loop
# --------------------------------------------------------------------------- #


async def test_a_transient_failure_is_retried_then_succeeds(unlimited):
    attempts = []

    async def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ApiError(503)
        return "ok"

    assert await gemini_limits.with_retries(flaky) == "ok"
    assert len(attempts) == 3


async def test_an_unretryable_failure_fails_on_the_first_attempt(unlimited):
    attempts = []

    async def bad_request():
        attempts.append(1)
        raise ApiError(400)

    with pytest.raises(ApiError):
        await gemini_limits.with_retries(bad_request, attempts=5)
    assert len(attempts) == 1


async def test_retries_stop_after_the_attempt_limit(unlimited):
    attempts = []

    async def always_overloaded():
        attempts.append(1)
        raise ApiError(503)

    with pytest.raises(ApiError):
        await gemini_limits.with_retries(always_overloaded, attempts=3)
    assert len(attempts) == 3


async def test_backoff_grows_and_is_jittered(no_sleep, unlimited):
    """Jitter matters: stage 2 fans out, so retries would otherwise sync up."""

    async def always_overloaded():
        raise ApiError(503)

    with pytest.raises(ApiError):
        await gemini_limits.with_retries(always_overloaded, attempts=4)

    assert len(no_sleep) == 3
    assert no_sleep == sorted(no_sleep)  # growing
    # Jitter means never exactly the base delay.
    assert all(delay != gemini_limits.RETRY_BASE_DELAY_SEC for delay in no_sleep[1:])


async def test_retry_after_from_the_server_wins_over_our_guess(no_sleep, unlimited):
    attempts = []

    async def rate_limited():
        attempts.append(1)
        if len(attempts) < 2:
            raise ApiError(429, retry_after="17")
        return "ok"

    assert await gemini_limits.with_retries(rate_limited) == "ok"
    assert no_sleep == [17.0]


async def test_backoff_is_capped(no_sleep, monkeypatch, unlimited):
    monkeypatch.setattr(gemini_limits, "RETRY_BASE_DELAY_SEC", 1_000_000.0)

    async def always_overloaded():
        raise ApiError(503)

    with pytest.raises(ApiError):
        await gemini_limits.with_retries(always_overloaded, attempts=2)

    assert no_sleep == [gemini_limits.RETRY_MAX_DELAY_SEC]


# --------------------------------------------------------------------------- #
# the token bucket
# --------------------------------------------------------------------------- #


async def test_acquire_returns_immediately_when_tokens_are_available(monkeypatch):
    async def _take(tokens=1):
        return 0.0

    monkeypatch.setattr(gemini_limits, "_take", _take)
    await gemini_limits.acquire()  # must not raise or sleep


async def test_acquire_waits_then_proceeds(monkeypatch, no_sleep):
    waits = [2.0, 0.0]

    async def _take(tokens=1):
        return waits.pop(0)

    monkeypatch.setattr(gemini_limits, "_take", _take)
    await gemini_limits.acquire()

    assert len(no_sleep) == 1
    assert no_sleep[0] >= 2.0  # jittered upward, never shorter than told


async def test_acquire_gives_up_rather_than_hanging_forever(monkeypatch, no_sleep):
    """A wedged bucket must surface as a stage failure, not a stuck worker."""

    async def _take(tokens=1):
        return 1000.0

    monkeypatch.setattr(gemini_limits, "_take", _take)

    with pytest.raises(TimeoutError):
        await gemini_limits.acquire(timeout_sec=10)


async def test_the_limiter_fails_open_when_redis_is_down(monkeypatch, caplog):
    """Throttling is a cost control; failing every job over it is worse."""

    async def _take(tokens=1):
        raise ConnectionError("redis is down")

    monkeypatch.setattr(gemini_limits, "_take", _take)

    with caplog.at_level("WARNING"):
        await gemini_limits.acquire()  # must not raise

    assert "limiter unavailable" in caplog.text


async def test_the_bucket_script_refills_over_time():
    """The Lua is the contract: refill, then take, atomically."""
    script = gemini_limits._TAKE_SCRIPT
    assert "HMGET" in script and "HMSET" in script
    # An idle bucket expires, so a fresh one legitimately starts full.
    assert "EXPIRE" in script
    assert "math.min(capacity" in script


def test_asyncio_timeout_is_treated_as_transient():
    assert gemini_limits.is_retryable(TimeoutError())
