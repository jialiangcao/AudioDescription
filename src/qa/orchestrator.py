"""QASystem: the cycle loop over the shared history blackboard.

Port of Symphony's VideoUnderstandingSystem. Each cycle: the CoreAgent plans
(one Gemini call), the chosen worker agent runs (its tool results come back as
text), and the record lands in `history`, which is re-fed to the planner. A
`finish` decision passes through the ReflectionAgent exactly once per question
(the `if_reflected` latch); afterwards any `finish` is accepted as-is.

Two Symphony crash paths are fixed here (an unparseable planner decision, and
worker agents returning None); everything else — record shapes, the latch, the
verbatim not-credible nudge sentence — is kept as in the original.
"""

import logging
from typing import Literal

from pydantic import BaseModel

from qa.config import MAX_CYCLES
from qa.core_agent import CoreAgent
from qa.frame_index import FrameIndex
from qa.llm import ToolContext
from qa.localize_agent import LocalizeAgent
from qa.perception_agent import PerceptionAgent
from qa.reflection_agent import ReflectionAgent
from qa.subtitle_agent import SubtitleAgent
from timeline import Timeline

logger = logging.getLogger(__name__)

NOT_CREDIBLE_ACTION = (
    "Upon reflection, your current answer is not reliable. Please reconsider "
    "carefully and reorganize the relevant information to provide a credible "
    "response."
)


class QAResult(BaseModel):
    status: Literal["completed", "failed"]
    answer: str | None
    reason: str
    cycles: int
    history: list[dict]


class QASystem:
    def __init__(
        self,
        timeline: Timeline,
        question: str,
        client,
        blobs,
        frame_index: FrameIndex | None = None,
        max_cycles: int = MAX_CYCLES,
    ):
        self.question = question
        self.video_duration = timeline.duration_sec
        self.max_cycles = max_cycles

        self.if_reflected = 0
        self.cycle_count = 0
        self.completed = False
        self.final_answer: str | None = None
        self.history: list[dict] = []

        frame_index = frame_index or FrameIndex.from_timeline(timeline)
        ctx = ToolContext(client=client, frame_index=frame_index, blobs=blobs)
        self.core_agent = CoreAgent(
            client, question=question, video_duration_sec=self.video_duration
        )
        self.perception_agent = PerceptionAgent(client, ctx=ctx)
        self.subtitle_agent = SubtitleAgent(
            client, question=question, timeline=timeline
        )
        self.localize_agent = LocalizeAgent(
            client,
            question=question,
            video_duration_sec=self.video_duration,
            ctx=ctx,
        )
        self.reflection_agent = ReflectionAgent(client, question=question)

    async def run(self) -> QAResult:
        while not self.completed and self.cycle_count < self.max_cycles:
            self.cycle_count += 1
            logger.info("=== Q&A cycle %d ===", self.cycle_count)

            core_decision = await self.core_agent.run(history=self.history)
            if core_decision is None:
                logger.warning(
                    "cycle %d: CoreAgent output unparseable", self.cycle_count
                )
                self.history.append(
                    {
                        "action": "parse_failure",
                        "result": "CoreAgent returned unparseable output.",
                    }
                )
                continue
            logger.info("cycle %d: core decision: %s", self.cycle_count, core_decision)

            agent_to_call = core_decision.get("agent")

            if agent_to_call == "PerceptionAgent":
                result = await self.perception_agent.run(
                    instruct=core_decision.get("instruct"),
                    question=self.question,
                    video_duration=self.video_duration,
                )
            elif agent_to_call == "SubtitleAgent":
                result = await self.subtitle_agent.run()
            elif agent_to_call == "LocalizeAgent":
                result = await self.localize_agent.run()
            elif agent_to_call == "finish":
                proposed_answer = core_decision.get("answer")
                self.history.append(
                    {
                        "action": agent_to_call,
                        "reason": core_decision.get("reason"),
                        "answer": proposed_answer,
                    }
                )

                # Reflection fires at most once per question; later finishes
                # are accepted unconditionally (Symphony's if_reflected latch).
                if not self.if_reflected:
                    assessment = await self.reflection_agent.run(
                        proposed_answer=proposed_answer, history=self.history
                    )
                    self.if_reflected = 1
                else:
                    assessment = {"credible": True}

                if assessment.get("credible"):
                    logger.info("cycle %d: answer accepted", self.cycle_count)
                    self.completed = True
                    self.final_answer = proposed_answer
                    self.history.append(
                        {
                            "action": "reflection",
                            "assessment": "credible",
                            "proposed_answer": proposed_answer,
                        }
                    )
                    self.history.append(
                        {"action": "finish", "answer": self.final_answer}
                    )
                    break
                logger.info("cycle %d: answer rejected by reflection", self.cycle_count)
                self.history.append(
                    {
                        "action": NOT_CREDIBLE_ACTION,
                        "assessment": "not_credible",
                        "comment": assessment.get("comment", "No comment provided."),
                    }
                )
                continue
            else:
                logger.warning(
                    "cycle %d: unknown agent requested: %r",
                    self.cycle_count,
                    agent_to_call,
                )
                self.history.append(
                    {"action": "unknown_agent", "decision": core_decision}
                )
                continue

            record: dict = {"action": agent_to_call}
            if core_decision.get("instruct") is not None:
                record["instruct"] = core_decision.get("instruct")
            record["reason"] = core_decision.get("reason")
            record["result"] = result
            self.history.append(record)
            logger.debug("cycle %d result: %s", self.cycle_count, result)

        if self.completed:
            return self._build_final_result("completed", "Task finished successfully.")
        return self._build_final_result("failed", "Exceeded maximum cycles.")

    def _build_final_result(
        self, status: Literal["completed", "failed"], reason: str
    ) -> QAResult:
        return QAResult(
            status=status,
            answer=self.final_answer,
            reason=reason,
            cycles=self.cycle_count,
            history=self.history,
        )


async def answer_question(timeline: Timeline, question: str, client, blobs) -> QAResult:
    """Answer a free-form question about a processed video's timeline."""
    logger.info(
        "answer_question: question=%r over %d segment(s)",
        question,
        len(timeline.segments),
    )
    system = QASystem(timeline=timeline, question=question, client=client, blobs=blobs)
    result = await system.run()
    logger.info(
        "answer_question: status=%s cycles=%d answer=%r",
        result.status,
        result.cycles,
        result.answer,
    )
    return result
