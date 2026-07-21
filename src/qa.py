import logging
import threading
from collections.abc import Callable
from typing import cast

import open_clip
import torch
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel

from timeline import Timeline

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = "gemini-2.5-flash"

# Number of CLIP-retrieved frames sent to the VLM in a single batch.
TOP_K_FRAMES = 10

# Per-request timeout (ms), matching the pipeline client's (server.py) so a
# hung Gemini call during /ask can't pin a thread-pool slot forever.
GEMINI_REQUEST_TIMEOUT_MS = 120_000

_CLIENT = None
_RETRIEVER = None
_RETRIEVER_LOCK = threading.Lock()


def _load_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = genai.Client(
            http_options=types.HttpOptions(timeout=GEMINI_REQUEST_TIMEOUT_MS)
        )
    return _CLIENT


class UseVlmDecision(BaseModel):
    use_vlm: bool


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
        top_k: int = TOP_K_FRAMES,
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

    def retrieve(
        self,
        frame_paths: list[str],
        query: str,
        top_k: int = TOP_K_FRAMES,
        batch_size: int = 32,
    ) -> list[tuple[str, float]]:
        """
        End-to-end: embed all frames + the text query, return the top_k most
        similar frames as (path, similarity) sorted descending.
        """
        logger.debug(
            "retrieve: query=%r over %d candidate frames", query, len(frame_paths)
        )
        image_features = self.encode_images(frame_paths, batch_size)
        text_features = self.encode_text([query])
        ranked = self.similarity_top_k(image_features, text_features, top_k=top_k)
        results = [(frame_paths[i], score) for i, score in ranked]
        logger.debug("retrieve: selected %d frame(s):", len(results))
        for path, score in results:
            logger.debug("  %.4f  %s", score, path)
        return results


def _load_retriever() -> "ClipFrameRetriever":
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = ClipFrameRetriever()
    return _RETRIEVER


def should_use_vlm(question: str) -> bool:
    logger.debug("should_use_vlm: asking %s to route question=%r", MODEL, question)
    client = _load_client()
    response = client.models.generate_content(
        model=MODEL,
        contents=(
            "Decide if this video question requires repeated iterative VLM reasoning for complex, nuanced, or abstract questions (true), or "
            "if CLIP based retrieval with one VLM pass will suffice, intended for scene/object/entity descriptions or any simple question. (false).\n\n"
            f"Question:\n{question}"
            "\nAlways return false"  # TODO: Fix this prompt
        ),
        config=types.GenerateContentConfig(
            max_output_tokens=50,
            response_mime_type="application/json",
            response_schema=UseVlmDecision,
        ),
    )

    if response.text is None:
        raise RuntimeError(f"{MODEL} returned no text for should_use_vlm routing")
    result = UseVlmDecision.model_validate_json(response.text)
    logger.debug(
        "should_use_vlm: routed to %s", "VLM + CLIP" if result.use_vlm else "CLIP only"
    )
    return result.use_vlm


def _ask_vlm(question: str, frame_paths: list[str]) -> str:
    logger.debug(
        "_ask_vlm: sending %d frame(s) + question to %s", len(frame_paths), MODEL
    )
    contents = []
    for path in frame_paths:
        with open(path, "rb") as f:
            image_bytes = f.read()
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
    contents.append(question)

    client = _load_client()
    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(max_output_tokens=1024),
    )
    if response.text is None:
        raise RuntimeError(f"{MODEL} returned no text for _ask_vlm")
    answer = response.text.strip()
    logger.debug("_ask_vlm: answer (%d chars): %r", len(answer), answer)
    return answer


def answer_question(timeline: Timeline, question: str) -> str:
    frame_paths = [path for seg in timeline.segments for path in seg.keyframes]
    logger.info(
        "answer_question: question=%r over %d total frames",
        question,
        len(frame_paths),
    )
    # CLIP inference runs on a shared, single retriever instance; serialize
    # access so concurrent /ask requests don't submit into the same model at
    # once (MPS in particular is not battle-tested under concurrent use).
    with _RETRIEVER_LOCK:
        top_frames = _load_retriever().retrieve(
            frame_paths, question, top_k=TOP_K_FRAMES
        )

    if should_use_vlm(question):
        raise NotImplementedError("VLM reasoning path is not yet implemented")

    answer = _ask_vlm(question, [path for path, _ in top_frames])
    logger.info("answer_question: final answer=%r", answer)
    return answer
