from types import SimpleNamespace
from typing import cast

from conftest import FakeGeminiClient

from qa.core_agent import CoreAgent
from qa.frame_index import FrameIndex
from qa.localize_agent import LocalizeAgent
from qa.orchestrator import NOT_CREDIBLE_ACTION, QASystem, answer_question
from qa.perception_agent import PerceptionAgent
from qa.reflection_agent import ReflectionAgent
from qa.subtitle_agent import SubtitleAgent
from timeline import Frame, Segment, Timeline


def _timeline():
    return Timeline(
        video_id="v.mp4",
        duration_sec=10.0,
        segments=[
            Segment(
                id=0,
                start=0.0,
                end=10.0,
                frames=[Frame(index=0, time=0.0, path="f.jpg")],
            )
        ],
    )


class FakeCore:
    def __init__(self, decisions):
        self.decisions = list(decisions)

    async def run(self, history):
        return self.decisions.pop(0)


class FakeWorker:
    def __init__(self, result="worker result"):
        self.result = result
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class FakeReflection:
    def __init__(self, assessments):
        self.assessments = list(assessments)
        self.calls = []

    async def run(self, proposed_answer, history):
        self.calls.append(proposed_answer)
        return self.assessments.pop(0)


def _system(decisions, assessments=({"credible": True},), max_cycles=17):
    """A QASystem with every agent replaced by a scripted fake.

    Returns (system, fakes); assert call records on the fakes (the system's
    attributes are typed to the real agent classes, hence the casts).
    """
    system = QASystem(
        timeline=_timeline(),
        question="q?",
        client=FakeGeminiClient(),
        frame_index=FrameIndex(entries=[(0.0, "f.jpg")], duration_sec=10.0),
        max_cycles=max_cycles,
    )
    fakes = SimpleNamespace(
        core=FakeCore(decisions),
        perception=FakeWorker("perceived"),
        subtitle=FakeWorker("subtitled"),
        localize=FakeWorker("localized"),
        reflection=FakeReflection(assessments),
    )
    system.core_agent = cast(CoreAgent, fakes.core)
    system.perception_agent = cast(PerceptionAgent, fakes.perception)
    system.subtitle_agent = cast(SubtitleAgent, fakes.subtitle)
    system.localize_agent = cast(LocalizeAgent, fakes.localize)
    system.reflection_agent = cast(ReflectionAgent, fakes.reflection)
    return system, fakes


async def test_dispatch_and_history_record_shapes():
    system, fakes = _system(
        [
            {
                "reason": "look",
                "agent": "PerceptionAgent",
                "instruct": "check [00:00:01, 00:00:09]",
            },
            {"reason": "dialogue", "agent": "SubtitleAgent"},
            {"reason": "ground", "agent": "LocalizeAgent"},
            {"reason": "done", "agent": "finish", "answer": "42"},
        ]
    )
    result = await system.run()

    assert result.status == "completed"
    assert result.answer == "42"
    assert result.cycles == 4
    # Perception records carry the instruct; the others don't.
    assert result.history[0] == {
        "action": "PerceptionAgent",
        "instruct": "check [00:00:01, 00:00:09]",
        "reason": "look",
        "result": "perceived",
    }
    assert result.history[1] == {
        "action": "SubtitleAgent",
        "reason": "dialogue",
        "result": "subtitled",
    }
    assert result.history[2] == {
        "action": "LocalizeAgent",
        "reason": "ground",
        "result": "localized",
    }
    # The finish tail: finish decision, credible reflection, final finish.
    assert result.history[3] == {"action": "finish", "reason": "done", "answer": "42"}
    assert result.history[4] == {
        "action": "reflection",
        "assessment": "credible",
        "proposed_answer": "42",
    }
    assert result.history[5] == {"action": "finish", "answer": "42"}
    # The instruct was forwarded to the perception agent.
    assert fakes.perception.calls[0]["instruct"] == "check [00:00:01, 00:00:09]"


async def test_reflection_rejection_continues_and_latches():
    system, fakes = _system(
        [
            {"reason": "r1", "agent": "finish", "answer": "first"},
            {"reason": "r2", "agent": "finish", "answer": "second"},
        ],
        assessments=[{"credible": False, "comment": "too hasty"}],
    )
    result = await system.run()

    assert result.status == "completed"
    assert result.answer == "second"
    # Reflection ran exactly once (the latch): the second finish auto-accepted.
    assert fakes.reflection.calls == ["first"]
    nudge = result.history[1]
    assert nudge["action"] == NOT_CREDIBLE_ACTION
    assert nudge["assessment"] == "not_credible"
    assert nudge["comment"] == "too hasty"


async def test_max_cycles_exhaustion_fails():
    system, fakes = _system(
        [{"reason": "r", "agent": "SubtitleAgent"}] * 3,
        max_cycles=3,
    )
    result = await system.run()
    assert result.status == "failed"
    assert result.answer is None
    assert result.reason == "Exceeded maximum cycles."
    assert result.cycles == 3


async def test_unparseable_core_decision_is_recorded_and_skipped():
    system, fakes = _system(
        [None, {"reason": "done", "agent": "finish", "answer": "ok"}],
    )
    result = await system.run()
    assert result.status == "completed"
    assert result.history[0] == {
        "action": "parse_failure",
        "result": "CoreAgent returned unparseable output.",
    }


async def test_unknown_agent_is_recorded_and_skipped():
    decision = {"reason": "r", "agent": "TimeTravelAgent"}
    system, fakes = _system(
        [decision, {"reason": "d", "agent": "finish", "answer": "ok"}]
    )
    result = await system.run()
    assert result.status == "completed"
    assert result.history[0] == {"action": "unknown_agent", "decision": decision}


async def test_answer_question_entrypoint(monkeypatch):
    # The public wrapper builds the FrameIndex and runs the system end-to-end;
    # with the canned client the planner immediately finishes and reflection
    # approves.
    result = await answer_question(_timeline(), "q?", FakeGeminiClient())
    assert result.status == "completed"
    assert result.answer == "a canned answer"
    assert result.history[-1] == {"action": "finish", "answer": "a canned answer"}
