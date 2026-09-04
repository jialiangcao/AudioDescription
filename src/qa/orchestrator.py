"""QASystem: the cycle loop over the shared history blackboard, as a LangGraph.

Port of Symphony's VideoUnderstandingSystem. Each cycle: the CoreAgent plans
(one Gemini call), the chosen worker agent runs (its tool results come back as
text), and the record lands in `history`, which is re-fed to the planner. A
`finish` decision passes through the ReflectionAgent exactly once per question
(the `if_reflected` latch); afterwards any `finish` is accepted as-is.

The control flow is a `StateGraph` rather than a `while` loop. What that buys is
not the loop itself — it is that every transition is a named node with typed
state, so a run can be streamed step by step to the browser (the worker
publishes each node's output as a `qa_trace` event) and checkpointed. `history`
maps onto a reducer-appended channel exactly as it was: nodes return only the
records they add.

The observable behaviour is unchanged and deliberately so: record shapes and
their key order, the latch, the verbatim not-credible nudge sentence, and the
two Symphony crash paths that were already fixed here (an unparseable planner
decision, and worker agents returning None). tests/test_qa_orchestrator.py is
the specification.
"""

import logging
import operator
from collections.abc import Awaitable, Callable
from typing import Annotated, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
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

# Awaited with (node_name, records_appended) as each graph node completes.
StepCallback = Callable[[str, list[dict]], Awaitable[None]]
# Awaited with the planner's decision the moment it is made — i.e. *before* the
# agent it names has run. `on_step` can only report a step once it has finished,
# and a worker agent runs for tens of seconds, so this is what lets a caller say
# what the run is doing right now rather than what it last did.
DecisionCallback = Callable[[dict], Awaitable[None]]

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


class QAState(TypedDict):
    """The blackboard every node reads and appends to.

    ``history`` is the only accumulating channel: nodes return just their new
    records and the reducer concatenates, which is what the loop used to do by
    mutating one shared list.
    """

    history: Annotated[list[dict], operator.add]
    cycle_count: int
    if_reflected: bool
    decision: dict | None
    proposed_answer: str | None
    final_answer: str | None
    completed: bool


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

    # -- nodes ---------------------------------------------------------------

    async def _plan(self, state: QAState) -> dict:
        """One planner call. Records a parse failure rather than crashing."""
        cycle = state["cycle_count"] + 1
        logger.info("=== Q&A cycle %d ===", cycle)

        decision = await self.core_agent.run(history=state["history"])
        if decision is None:
            logger.warning("cycle %d: CoreAgent output unparseable", cycle)
            return {
                "cycle_count": cycle,
                "decision": None,
                "history": [
                    {
                        "action": "parse_failure",
                        "result": "CoreAgent returned unparseable output.",
                    }
                ],
            }
        logger.info("cycle %d: core decision: %s", cycle, decision)
        return {"cycle_count": cycle, "decision": decision}

    def _worker_record(self, decision: dict, result) -> dict:
        """The history entry for a worker call.

        Key insertion order matters: history is serialized straight into the
        planner's next prompt, so reordering these changes the model's input.
        """
        record: dict = {"action": decision.get("agent")}
        if decision.get("instruct") is not None:
            record["instruct"] = decision.get("instruct")
        record["reason"] = decision.get("reason")
        record["result"] = result
        return record

    async def _perception(self, state: QAState) -> dict:
        decision = state["decision"] or {}
        result = await self.perception_agent.run(
            instruct=decision.get("instruct"),
            question=self.question,
            video_duration=self.video_duration,
        )
        logger.debug("cycle %d result: %s", state["cycle_count"], result)
        return {"history": [self._worker_record(decision, result)]}

    async def _subtitle(self, state: QAState) -> dict:
        decision = state["decision"] or {}
        result = await self.subtitle_agent.run()
        logger.debug("cycle %d result: %s", state["cycle_count"], result)
        return {"history": [self._worker_record(decision, result)]}

    async def _localize(self, state: QAState) -> dict:
        decision = state["decision"] or {}
        result = await self.localize_agent.run()
        logger.debug("cycle %d result: %s", state["cycle_count"], result)
        return {"history": [self._worker_record(decision, result)]}

    async def _unknown_agent(self, state: QAState) -> dict:
        decision = state["decision"] or {}
        logger.warning(
            "cycle %d: unknown agent requested: %r",
            state["cycle_count"],
            decision.get("agent"),
        )
        return {"history": [{"action": "unknown_agent", "decision": decision}]}

    async def _finish(self, state: QAState) -> dict:
        """A proposed answer, screened by reflection at most once per question."""
        decision = state["decision"] or {}
        proposed = decision.get("answer")
        cycle = state["cycle_count"]

        # The finish record lands *before* reflecting, so the critic sees it.
        records: list[dict] = [
            {
                "action": "finish",
                "reason": decision.get("reason"),
                "answer": proposed,
            }
        ]

        # Reflection fires at most once per question; later finishes are
        # accepted unconditionally (Symphony's if_reflected latch).
        if not state["if_reflected"]:
            assessment = await self.reflection_agent.run(
                proposed_answer=proposed,
                history=state["history"] + records,
            )
        else:
            assessment = {"credible": True}

        if assessment.get("credible"):
            logger.info("cycle %d: answer accepted", cycle)
            records.append(
                {
                    "action": "reflection",
                    "assessment": "credible",
                    "proposed_answer": proposed,
                }
            )
            records.append({"action": "finish", "answer": proposed})
            return {
                "history": records,
                "if_reflected": True,
                "completed": True,
                "final_answer": proposed,
                "proposed_answer": proposed,
            }

        logger.info("cycle %d: answer rejected by reflection", cycle)
        records.append(
            {
                "action": NOT_CREDIBLE_ACTION,
                "assessment": "not_credible",
                "comment": assessment.get("comment", "No comment provided."),
            }
        )
        return {"history": records, "if_reflected": True, "proposed_answer": proposed}

    # -- routing -------------------------------------------------------------

    def _route_from_plan(self, state: QAState) -> str:
        if state["cycle_count"] >= self.max_cycles and state["decision"] is None:
            return END
        decision = state["decision"]
        if decision is None:
            return "plan"  # parse failure: already recorded, try again
        agent = decision.get("agent")
        if agent == "PerceptionAgent":
            return "perception"
        if agent == "SubtitleAgent":
            return "subtitle"
        if agent == "LocalizeAgent":
            return "localize"
        if agent == "finish":
            return "finish"
        return "unknown_agent"

    def _route_after_step(self, state: QAState) -> str:
        """Back to the planner unless the answer stuck or we ran out of cycles."""
        if state["completed"]:
            return END
        if state["cycle_count"] >= self.max_cycles:
            return END
        return "plan"

    def build_graph(self):
        """Compile the graph.

        Built per run rather than once at construction, so swapping an agent on
        the instance (as the tests do) is picked up — the nodes are bound
        methods that read ``self`` when they execute.
        """
        graph = StateGraph(QAState)
        graph.add_node("plan", self._plan)
        graph.add_node("perception", self._perception)
        graph.add_node("subtitle", self._subtitle)
        graph.add_node("localize", self._localize)
        graph.add_node("unknown_agent", self._unknown_agent)
        graph.add_node("finish", self._finish)

        graph.set_entry_point("plan")
        graph.add_conditional_edges(
            "plan",
            self._route_from_plan,
            {
                "plan": "plan",
                "perception": "perception",
                "subtitle": "subtitle",
                "localize": "localize",
                "finish": "finish",
                "unknown_agent": "unknown_agent",
                END: END,
            },
        )
        for node in ("perception", "subtitle", "localize", "unknown_agent", "finish"):
            graph.add_conditional_edges(
                node, self._route_after_step, {"plan": "plan", END: END}
            )
        return graph.compile()

    # -- entry point ---------------------------------------------------------

    async def run(
        self,
        on_step: StepCallback | None = None,
        on_decision: DecisionCallback | None = None,
    ) -> QAResult:
        """Run the graph to completion.

        ``on_step`` (if given) is awaited with each node's name and the records
        it just appended, as they happen — which is the point of the graph:
        the browser can watch the reasoning trace build instead of waiting
        minutes for the finished answer.

        ``on_decision`` is awaited with each planner decision as soon as it is
        made, which is the *only* forward-looking signal in the run: it names
        the agent about to spend the next tens of seconds working, where
        ``on_step`` can only report that agent once it has already finished.
        """
        initial: QAState = {
            "history": [],
            "cycle_count": 0,
            "if_reflected": False,
            "decision": None,
            "proposed_answer": None,
            "final_answer": None,
            "completed": False,
        }
        # Each cycle is at most two super-steps; the real stopping condition is
        # max_cycles, checked in the routers. This is only a backstop against a
        # graph that somehow never terminates.
        graph = self.build_graph()
        config: RunnableConfig = {"recursion_limit": self.max_cycles * 4 + 10}

        if on_step is None and on_decision is None:
            final = await graph.ainvoke(initial, config=config)
        else:
            final = await self._stream(graph, initial, config, on_step, on_decision)

        if final["completed"]:
            return self._build_final_result(
                final, "completed", "Task finished successfully."
            )
        return self._build_final_result(final, "failed", "Exceeded maximum cycles.")

    async def _stream(
        self,
        graph,
        initial,
        config,
        on_step: StepCallback | None,
        on_decision: DecisionCallback | None = None,
    ) -> QAState:
        """Run the graph, reporting each node's updates, and rebuild the state.

        ``stream_mode="updates"`` yields only what each node returned, so the
        accumulated state has to be folded here — mirroring the reducers, which
        for this graph means concatenating ``history`` and overwriting the rest.
        """
        state: QAState = dict(initial)  # type: ignore[assignment]
        async for chunk in graph.astream(initial, config=config, stream_mode="updates"):
            for node, update in chunk.items():
                if not update:
                    continue
                for key, value in update.items():
                    if key == "history":
                        state["history"] = state["history"] + value
                    else:
                        state[key] = value  # type: ignore[literal-required]
                if on_step is not None:
                    await on_step(node, update.get("history", []))
                # After the planner, and only then, the next agent is known.
                if on_decision is not None and node == "plan" and state["decision"]:
                    await on_decision(state["decision"])
        return state

    def _build_final_result(
        self, state, status: Literal["completed", "failed"], reason: str
    ) -> QAResult:
        return QAResult(
            status=status,
            answer=state["final_answer"],
            reason=reason,
            cycles=state["cycle_count"],
            history=state["history"],
        )


async def answer_question(
    timeline: Timeline, question: str, client, blobs, on_step=None, on_decision=None
) -> QAResult:
    """Answer a free-form question about a processed video's timeline.

    ``on_step`` (optional) is awaited with each agent step as it completes, so
    a caller can stream the reasoning trace rather than wait for the answer.
    ``on_decision`` (optional) is awaited with each planner decision as it is
    made, naming the agent that is *about* to run.
    """
    logger.info(
        "answer_question: question=%r over %d segment(s)",
        question,
        len(timeline.segments),
    )
    system = QASystem(timeline=timeline, question=question, client=client, blobs=blobs)
    result = await system.run(on_step=on_step, on_decision=on_decision)
    logger.info(
        "answer_question: status=%s cycles=%d answer=%r",
        result.status,
        result.cycles,
        result.answer,
    )
    return result
