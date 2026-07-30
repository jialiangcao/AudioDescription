"""CLIP frame retrieval (open_clip) with a process-wide embedding cache.

``ClipFrameRetriever`` is moved unchanged from the pre-package qa.py; it fills
the role LanguageBind played in Symphony. New here: a path-keyed embedding
cache so repeated tool calls over the same job's frames only pay the image
encoding cost once, and an async entry point that runs the (blocking, GPU/MPS)
work in a thread under the shared lock.
"""

import asyncio
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import cast

import open_clip
import torch
from PIL import Image

from qa.config import MAX_CACHED_EMBEDDINGS

logger = logging.getLogger(__name__)

_RETRIEVER = None
_RETRIEVER_LOCK = threading.Lock()

# abs frame path -> 1D L2-normalized embedding row (CPU). FIFO-evicted; guarded
# by _RETRIEVER_LOCK together with the model itself.
_EMBED_CACHE: OrderedDict[str, torch.Tensor] = OrderedDict()


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("auto-selected device: %s", device)
    return device


class ClipFrameRetriever:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ):
        self.device = torch.device(device) if device else _pick_device()
        logger.info("loading CLIP model %r (pretrained=%r)...", model_name, pretrained)
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        # create_model_and_transforms is typed generically (plain nn.Module / a
        # preprocess union that includes a non-callable tuple variant); we know
        # concretely which types come back for a CLIP checkpoint like this one.
        self.model = cast(open_clip.CLIP, model).to(self.device).eval()
        self.preprocess = cast(Callable[[Image.Image], torch.Tensor], preprocess)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        logger.info("CLIP model ready on %s", self.device)

    @torch.no_grad()
    def encode_images(
        self, image_paths: list[str], batch_size: int = 32
    ) -> torch.Tensor:
        """Returns L2-normalized image embeddings, shape (N, D)."""
        if not image_paths:
            logger.debug("encode_images: no paths given, skipping")
            return torch.empty(0)

        logger.debug(
            "encode_images: embedding %d frames (batch_size=%d)",
            len(image_paths),
            batch_size,
        )
        all_feats = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            logger.debug("  batch %d: %d frames", i // batch_size, len(batch_paths))
            images = torch.stack(
                [self.preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            ).to(self.device)
            feats = self.model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu())

        result = torch.cat(all_feats, dim=0)
        logger.debug("encode_images: done, embeddings shape=%s", tuple(result.shape))
        return result

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Returns L2-normalized text embeddings, shape (N, D)."""
        logger.debug("encode_text: embedding %d text(s): %r", len(texts), texts)
        tokens = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu()

    def similarity_top_k(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        top_k: int,
    ) -> list[tuple[int, float]]:
        """
        Cosine similarity (both inputs must already be L2-normalized) between
        a single text query and all image features. Returns
        [(frame_index, similarity), ...] sorted descending, length min(top_k, N).
        """
        if image_features.nelement() == 0 or text_features.nelement() == 0:
            logger.debug("similarity_top_k: empty features, returning no matches")
            return []

        k = min(top_k, image_features.shape[0])
        similarity = (image_features @ text_features.T).squeeze(-1)  # (N,)
        topk_values, topk_indices = torch.topk(similarity, k=k, largest=True)
        ranked = [
            (int(idx.item()), float(val.item()))
            for idx, val in zip(topk_indices, topk_values, strict=True)
        ]
        logger.debug(
            "similarity_top_k: top-%d of %d frames, scores=%s",
            k,
            image_features.shape[0],
            [round(s, 4) for _, s in ranked],
        )
        return ranked


def _load_retriever() -> "ClipFrameRetriever":
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = ClipFrameRetriever()
    return _RETRIEVER


def _cached_image_features(
    retriever: ClipFrameRetriever, frame_paths: list[str]
) -> torch.Tensor:
    """Image embeddings for frame_paths (in order), encoding only cache misses."""
    uncached = [p for p in frame_paths if p not in _EMBED_CACHE]
    if uncached:
        feats = retriever.encode_images(uncached)
        for path, row in zip(uncached, feats, strict=True):
            _EMBED_CACHE[path] = row
        while len(_EMBED_CACHE) > MAX_CACHED_EMBEDDINGS:
            _EMBED_CACHE.popitem(last=False)
        logger.debug(
            "embed cache: %d new, %d total",
            len(uncached),
            len(_EMBED_CACHE),
        )
    # A cache miss above may itself have been evicted if frame_paths exceeds
    # the cache bound; fall back to re-encoding those rows individually.
    rows = []
    for path in frame_paths:
        row = _EMBED_CACHE.get(path)
        if row is None:
            row = retriever.encode_images([path])[0]
        rows.append(row)
    return torch.stack(rows)


def _retrieve_cached(
    frame_paths: list[str], cue: str, top_k: int
) -> list[tuple[str, float]]:
    if not frame_paths:
        return []
    # CLIP inference runs on a shared, single retriever instance; serialize
    # access so concurrent tool calls don't submit into the same model at once
    # (MPS in particular is not battle-tested under concurrent use).
    with _RETRIEVER_LOCK:
        retriever = _load_retriever()
        image_features = _cached_image_features(retriever, frame_paths)
        text_features = retriever.encode_text([cue])
        ranked = retriever.similarity_top_k(image_features, text_features, top_k)
    results = [(frame_paths[i], score) for i, score in ranked]
    logger.debug("retrieve: cue=%r -> %d frame(s)", cue, len(results))
    return results


async def retrieve_top_k(
    frame_paths: list[str], cue: str, top_k: int
) -> list[tuple[str, float]]:
    """Top-k frames for a text cue, as (path, similarity) sorted descending."""
    return await asyncio.to_thread(_retrieve_cached, frame_paths, cue, top_k)
