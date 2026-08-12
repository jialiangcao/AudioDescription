import json

import pytest

import events


@pytest.fixture(autouse=True)
def _reset_client():
    events._client = None
    yield
    events._client = None


class FakeRedis:
    def __init__(self, fail=False):
        self.published = []
        self.fail = fail

    async def publish(self, channel, message):
        if self.fail:
            raise ConnectionError("redis is down")
        self.published.append((channel, message))


def test_channel_is_namespaced_per_job():
    assert events.channel("abc") == "job:abc:events"


async def test_publish_sends_json_on_the_jobs_channel(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(events, "get_redis", lambda: fake)

    await events.publish("job-1", {"type": "stage", "stage": "vision", "seq": 7})

    (channel, message) = fake.published[0]
    assert channel == "job:job-1:events"
    assert json.loads(message) == {"type": "stage", "stage": "vision", "seq": 7}


async def test_publish_swallows_redis_failures(monkeypatch, caplog):
    """Fan-out is best-effort: the event is already durable in job_events.

    A Redis outage must not fail the pipeline stage that produced the event —
    clients recover it by replaying from their last seq.
    """
    monkeypatch.setattr(events, "get_redis", lambda: FakeRedis(fail=True))

    await events.publish("job-1", {"type": "stage"})  # must not raise

    assert "publish failed" in caplog.text


def test_redis_url_defaults_to_localhost(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert events.redis_url() == "redis://localhost:6379/0"

    monkeypatch.setenv("REDIS_URL", "redis://cache.internal:6379/2")
    assert events.redis_url() == "redis://cache.internal:6379/2"
