import base64
import json
from collections.abc import Callable
from typing import cast

import anthropic
import open_clip
import torch
from anthropic.types import TextBlock
from dotenv import load_dotenv
from PIL import Image

from timeline import Timeline

load_dotenv()

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"

# Number of CLIP-retrieved frames sent to the VLM in a single batch.
TOP_K_FRAMES = 10

SHOULD_USE_VLM_SCHEMA = {
    "type": "object",
    "properties": {
        "use_vlm": {
            "type": "boolean",
            "description": "True if the question requires nuance/abstract/complicated thinking to solve, false if it involves localization/scene description/object/entity descriptions or retrieval",
        },
    },
    "required": ["use_vlm"],
    "additionalProperties": False,
}


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[qa] auto-selected device: {device}")
    return device


class ClipFrameRetriever:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ):
        self.device = torch.device(device) if device else _pick_device()
        print(f"[qa] loading CLIP model {model_name!r} (pretrained={pretrained!r})...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        # create_model_and_transforms is typed generically (plain nn.Module / a
        # preprocess union that includes a non-callable tuple variant); we know
        # concretely which types come back for a CLIP checkpoint like this one.
        self.model = cast(open_clip.CLIP, model).to(self.device).eval()
        self.preprocess = cast(Callable[[Image.Image], torch.Tensor], preprocess)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        print(f"[qa] CLIP model ready on {self.device}")

    @torch.no_grad()
    def encode_images(
        self, image_paths: list[str], batch_size: int = 32
    ) -> torch.Tensor:
        """Returns L2-normalized image embeddings, shape (N, D)."""
        if not image_paths:
            print("[qa] encode_images: no paths given, skipping")
            return torch.empty(0)

        print(
            f"[qa] encode_images: embedding {len(image_paths)} frames (batch_size={batch_size})"
        )
        all_feats = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            print(f"[qa]   batch {i // batch_size}: {len(batch_paths)} frames")
            images = torch.stack(
                [self.preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            ).to(self.device)
            feats = self.model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu())

        result = torch.cat(all_feats, dim=0)
        print(f"[qa] encode_images: done, embeddings shape={tuple(result.shape)}")
        return result

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Returns L2-normalized text embeddings, shape (N, D)."""
        print(f"[qa] encode_text: embedding {len(texts)} text(s): {texts!r}")
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
            print("[qa] similarity_top_k: empty features, returning no matches")
            return []

        k = min(top_k, image_features.shape[0])
        similarity = (image_features @ text_features.T).squeeze(-1)  # (N,)
        topk_values, topk_indices = torch.topk(similarity, k=k, largest=True)
        ranked = [
            (int(idx.item()), float(val.item()))
            for idx, val in zip(topk_indices, topk_values, strict=True)
        ]
        print(
            f"[qa] similarity_top_k: top-{k} of {image_features.shape[0]} frames, scores={[round(s, 4) for _, s in ranked]}"
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
        print(
            f"[qa] retrieve: query={query!r} over {len(frame_paths)} candidate frames"
        )
        image_features = self.encode_images(frame_paths, batch_size)
        text_features = self.encode_text([query])
        ranked = self.similarity_top_k(image_features, text_features, top_k=top_k)
        results = [(frame_paths[i], score) for i, score in ranked]
        print(f"[qa] retrieve: selected {len(results)} frame(s):")
        for path, score in results:
            print(f"[qa]   {score:.4f}  {path}")
        return results


retriever = ClipFrameRetriever()


def should_use_vlm(question: str) -> bool:
    print(f"[qa] should_use_vlm: asking {MODEL} to route question={question!r}")
    response = client.messages.create(
        model=MODEL,
        max_tokens=50,
        output_config={
            "format": {"type": "json_schema", "schema": SHOULD_USE_VLM_SCHEMA}
        },
        messages=[
            {
                "role": "user",
                "content": (
                    "Decide if this video question requires repeated iterative VLM reasoning for complex, nuanced, or abstract questions (true), or "
                    "if CLIP based retrieval with one VLM pass will suffice, intended for scene/object/entity descriptions or any simple question. (false).\n\n"
                    f"Question:\n{question}"
                    "\nAlways return false"  # TODO: Fix this prompt"
                ),
            }
        ],
    )

    block = response.content[0]
    if isinstance(block, TextBlock):
        result = json.loads(block.text)
        print(
            f"[qa] should_use_vlm: routed to {'VLM + CLIP' if result['use_vlm'] else 'CLIP only'}"
        )
        return result["use_vlm"]

    print(
        f"[qa] should_use_vlm: expected a text block, got {type(block).__name__}; defaulting to CLIP only"
    )
    return False


def _ask_vlm(question: str, frame_paths: list[str]) -> str:
    print(f"[qa] _ask_vlm: sending {len(frame_paths)} frame(s) + question to {MODEL}")
    content = []
    for path in frame_paths:
        with open(path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_data,
                },
            }
        )
    content.append({"type": "text", "text": question})

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )
    answer = next(b.text for b in response.content if b.type == "text").strip()
    print(f"[qa] _ask_vlm: answer ({len(answer)} chars): {answer!r}")
    return answer


def answer_question(timeline: Timeline, question: str) -> str:
    frame_paths = [path for seg in timeline.segments for path in seg.keyframes]
    print(
        f"[qa] answer_question: question={question!r} over {len(frame_paths)} total frames"
    )
    top_frames = retriever.retrieve(frame_paths, question, top_k=TOP_K_FRAMES)

    if should_use_vlm(question):
        raise NotImplementedError("VLM reasoning path is not yet implemented")

    answer = _ask_vlm(question, [path for path, _ in top_frames])
    print(f"[qa] answer_question: final answer={answer!r}")
    return answer
