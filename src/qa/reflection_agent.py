"""ReflectionAgent: one-shot critic of the planner's proposed answer.

Port of Symphony's ReflectionAgent, including its deliberate fail-open: if a
valid assessment cannot be obtained, the answer is treated as credible so the
orchestrator can't deadlock on a broken critic.
"""

import json
import logging

from pydantic import BaseModel

from qa.config import THINKING_PLANNER
from qa.llm import generate_text
from qa.prompts import REFLECTION_PROMPT, REFLECTION_SYSTEM_PROMPT
from qa.utils import fix_and_parse_json, with_retries

logger = logging.getLogger(__name__)

FAIL_OPEN_ASSESSMENT = {
    "credible": True,
    "comment": "Fallback: Could not get a valid reflection.",
}


class ReflectionAssessment(BaseModel):
    credible: bool
    comment: str | None = None


class ReflectionAgent:
    def __init__(self, client, question: str):
        self.client = client
        self.question = question

    async def run(self, proposed_answer: str | None, history: list[dict]) -> dict:
        history_str = "\n".join(json.dumps(h) for h in history)
        prompt = (
            REFLECTION_PROMPT.replace("HISTORY_PLACEHOLDER", history_str)
            .replace("QUESTION_PLACEHOLDER", self.question)
            .replace("PROPOSED_ANSWER_PLACEHOLDER", str(proposed_answer))
        )
        try:
            text = await with_retries(
                lambda: generate_text(
                    self.client,
                    system=REFLECTION_SYSTEM_PROMPT,
                    user=prompt,
                    schema=ReflectionAssessment,
                    thinking=THINKING_PLANNER,
                )
            )
        except Exception:
            logger.exception("ReflectionAgent call failed; failing open")
            return dict(FAIL_OPEN_ASSESSMENT)

        try:
            assessment = ReflectionAssessment.model_validate_json(text).model_dump()
        except ValueError:
            parsed = await fix_and_parse_json(text, self.client)
            if not (isinstance(parsed, dict) and "credible" in parsed):
                logger.warning("ReflectionAgent output unusable; failing open")
                return dict(FAIL_OPEN_ASSESSMENT)
            assessment = parsed
        logger.info("ReflectionAgent assessment: %s", assessment)
        return assessment
