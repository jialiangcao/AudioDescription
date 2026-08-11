import pytest
import torch

import qa.retriever as retriever_mod
from qa.retriever import retrieve_top_k


class FakeClipModel:
    """Stands in for ClipFrameRetriever; counts which paths get encoded."""

    def __init__(self):
        self.encoded = []

    def encode_images(self, image_paths, batch_size=32):
        self.encoded.extend(image_paths)
        return torch.ones(len(image_paths), 4)

    def encode_text(self, texts):
        return torch.ones(len(texts), 4)

    def similarity_top_k(self, image_features, text_features, top_k):
        k = min(top_k, image_features.shape[0])
        return [(i, 1.0 - i * 0.01) for i in range(k)]


@pytest.fixture
def fake_clip(monkeypatch):
    fake = FakeClipModel()
    monkeypatch.setattr(retriever_mod, "_load_retriever", lambda: fake)
    retriever_mod._EMBED_CACHE.clear()
    yield fake
    retriever_mod._EMBED_CACHE.clear()


def _keys(blobs, *names):
    """Blob keys backed by real (empty) files, so ``fetch`` can resolve them."""
    keys = []
    for name in names:
        key = f"frames/{name}"
        blobs.path(key).write_bytes(b"fake-jpeg")
        keys.append(key)
    return keys


async def test_second_retrieve_hits_the_cache(fake_clip, job_blobs):
    keys = _keys(job_blobs, *(f"{i}.jpg" for i in range(5)))

    first = await retrieve_top_k(keys, "a dog", 3, job_blobs)
    assert [k for k, _ in first] == keys[:3]
    assert len(fake_clip.encoded) == 5

    second = await retrieve_top_k(keys, "a cat", 3, job_blobs)
    assert [k for k, _ in second] == keys[:3]
    # No new image encodes: everything came from the cache.
    assert len(fake_clip.encoded) == 5


async def test_only_new_keys_are_encoded(fake_clip, job_blobs):
    a, b, c = _keys(job_blobs, "a.jpg", "b.jpg", "c.jpg")
    await retrieve_top_k([a, b], "x", 2, job_blobs)
    await retrieve_top_k([b, c], "x", 2, job_blobs)

    encoded = [p.rsplit("/", 1)[-1] for p in fake_clip.encoded]
    assert encoded == ["a.jpg", "b.jpg", "c.jpg"]


async def test_cache_is_keyed_per_job_so_identical_keys_do_not_collide(
    fake_clip, tmp_path
):
    """Job-relative keys repeat across jobs; the cache must not conflate them.

    Every job's frames are named frames/shot_0000_00.jpg, so a cache keyed on
    the job-relative key would serve job A's embedding for job B's frame.
    """
    from blobs import JobBlobs

    key = "frames/shot_0000_00.jpg"
    job_a = JobBlobs("job-a", root=tmp_path / "a")
    job_b = JobBlobs("job-b", root=tmp_path / "b")
    job_a.path(key).write_bytes(b"frame-from-a")
    job_b.path(key).write_bytes(b"frame-from-b")

    await retrieve_top_k([key], "x", 1, job_a)
    await retrieve_top_k([key], "x", 1, job_b)

    # Both frames were encoded, and both cache entries survive side by side.
    assert len(fake_clip.encoded) == 2
    assert set(retriever_mod._EMBED_CACHE) == {
        "jobs/job-a/" + key,
        "jobs/job-b/" + key,
    }


async def test_cache_eviction_is_bounded(fake_clip, monkeypatch, job_blobs):
    monkeypatch.setattr(retriever_mod, "MAX_CACHED_EMBEDDINGS", 2)
    keys = _keys(job_blobs, *(f"{i}.jpg" for i in range(4)))
    await retrieve_top_k(keys, "x", 2, job_blobs)
    assert len(retriever_mod._EMBED_CACHE) == 2  # FIFO-evicted to the bound


async def test_empty_keys_short_circuit(fake_clip, job_blobs):
    assert await retrieve_top_k([], "x", 5, job_blobs) == []
    assert fake_clip.encoded == []
