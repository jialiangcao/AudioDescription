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


async def test_second_retrieve_hits_the_cache(fake_clip):
    paths = [f"/f/{i}.jpg" for i in range(5)]

    first = await retrieve_top_k(paths, "a dog", top_k=3)
    assert [p for p, _ in first] == paths[:3]
    assert fake_clip.encoded == paths

    second = await retrieve_top_k(paths, "a cat", top_k=3)
    assert [p for p, _ in second] == paths[:3]
    # No new image encodes: everything came from the cache.
    assert fake_clip.encoded == paths


async def test_only_new_paths_are_encoded(fake_clip):
    await retrieve_top_k(["/f/a.jpg", "/f/b.jpg"], "x", top_k=2)
    await retrieve_top_k(["/f/b.jpg", "/f/c.jpg"], "x", top_k=2)
    assert fake_clip.encoded == ["/f/a.jpg", "/f/b.jpg", "/f/c.jpg"]


async def test_cache_eviction_is_bounded(fake_clip, monkeypatch):
    monkeypatch.setattr(retriever_mod, "MAX_CACHED_EMBEDDINGS", 2)
    paths = [f"/f/{i}.jpg" for i in range(4)]
    await retrieve_top_k(paths, "x", top_k=2)
    assert len(retriever_mod._EMBED_CACHE) == 2  # FIFO-evicted to the bound


async def test_empty_paths_short_circuit(fake_clip):
    assert await retrieve_top_k([], "x", top_k=5) == []
    assert fake_clip.encoded == []
