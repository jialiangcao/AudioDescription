"""CoreAgent: the text-only planner. One Gemini call per cycle -> one decision.

Port of Symphony's CoreAgent. The decision comes back as Gemini structured
output (CoreDecision schema) while the prompt keeps its JSON templates, so
fix_and_parse_json is only a backstop for malformed output.
"""

import logging

from pydantic import BaseModel

from qa.config import THINKING_PLANNER
from qa.llm import generate_text
from qa.prompts import CORE_SYSTEM_PROMPT, build_core_prompt
from qa.utils import convert_seconds_to_hhmmss, fix_and_parse_json, with_retries

logger = logging.getLogger(__name__)


class CoreDecision(BaseModel):
    reason: str
    agent: str
    instruct: str | None = None
    answer: str | None = None


class CoreAgent:
    def __init__(self, client, question: str, video_duration_sec: float):
        self.client = client
        self.question = question
        self.duration_str = convert_seconds_to_hhmmss(video_duration_sec)

    async def run(self, history: list[dict]) -> dict | None:
        """One planning step. Returns the decision dict, or None if the
        output could not be parsed even after LLM repair (the orchestrator
        records that and moves to the next cycle)."""
        prompt = build_core_prompt(self.question, history, self.duration_str)
        try:
            text = await with_retries(
                lambda: generate_text(
                    self.client,
                    system=CORE_SYSTEM_PROMPT,
                    user=prompt,
                    schema=CoreDecision,
                    thinking=THINKING_PLANNER,
                )
            )
        except Exception:
            logger.exception("CoreAgent call failed")
            return None

        try:
            return CoreDecision.model_validate_json(text).model_dump(exclude_none=True)
        except ValueError:
            logger.warning("CoreAgent output failed schema validation: %r", text)
            return await fix_and_parse_json(text, self.client)
