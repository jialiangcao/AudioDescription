"""PerceptionAgent: the multi-turn ReAct loop over the frame-inspection tools.

Port of Symphony's PerceptionAgent: up to PERCEPTION_MAX_ITERATIONS turns; the
final turn force-requests an answer; a turn that yields neither an `[answer]`
text nor tool calls is re-sampled up to NO_TOOL_CALL_RETRIES times. Every tool
call in a turn is executed and its result appended as a function response.
Always returns a string (Symphony could crash on None content or fall off the
loop returning None; both are guarded here).
"""

import logging

from google.genai import types

from qa.config import NO_TOOL_CALL_RETRIES, PERCEPTION_MAX_ITERATIONS
from qa.llm import (
    ToolContext,
    generate_with_tools,
    tool_response_content,
    user_content,
)
from qa.prompts import (
    FORCE_ANSWER_PROMPT,
    PERCEPTION_AGENT_PROMPT,
    PERCEPTION_SYSTEM_PROMPT,
)
from qa.tools_perception import (
    FRAME_ASSOCIATE_TOOL_DECLARATION,
    FRAME_INSPECT_TOOL_DECLARATION,
    INTERVAL_SUMMARY_TOOL_DECLARATION,
    frame_associate_tool,
    frame_inspect_tool,
    interval_summary_tool,
)
from qa.utils import with_retries

logger = logging.getLogger(__name__)


class PerceptionAgent:
    def __init__(
        self, client, ctx: ToolContext, max_iterations: int = PERCEPTION_MAX_ITERATIONS
    ):
        self.client = client
        self.ctx = ctx
        self.max_iterations = max_iterations
        self.tools = [
            types.Tool(
                function_declarations=[
                    FRAME_INSPECT_TOOL_DECLARATION,
                    INTERVAL_SUMMARY_TOOL_DECLARATION,
                    FRAME_ASSOCIATE_TOOL_DECLARATION,
                ]
            )
        ]

    async def run(
        self, instruct: str | None, question: str, video_duration: float
    ) -> str:
        # question/video_duration are accepted for interface parity with
        # Symphony, whose lv prompt (like this one) only injects the Instruct.
        del question, video_duration
        prompt = PERCEPTION_AGENT_PROMPT.replace("Instruct_PLACEHOLDER", str(instruct))
        contents: list[types.Content] = [user_content(prompt)]

        text = ""
        for iteration in range(self.max_iterations):
            if iteration == self.max_iterations - 1:
                contents.append(user_content(FORCE_ANSWER_PROMPT))

            response = None
            for _ in range(NO_TOOL_CALL_RETRIES):
                response = await with_retries(
                    lambda: generate_with_tools(
                        self.client,
                        system=PERCEPTION_SYSTEM_PROMPT,
                        contents=contents,
                        tools=self.tools,
                    )
                )
                text = response.text or ""
                if "[answer]" in text or response.function_calls:
                    break
                logger.info("PerceptionAgent: neither answer nor tool call, retrying")

            if "[answer]" in text:
                logger.debug("PerceptionAgent: answered on iteration %d", iteration + 1)
                return text
            if response is None or not response.function_calls:
                return text

            contents.append(response.candidates[0].content)
            parts = []
            for call in response.function_calls:
                result = await self._exec_tool(call)
                parts.append(
                    types.Part.from_function_response(
                        name=call.name or "", response={"result": str(result)}
                    )
                )
            contents.append(tool_response_content(parts))

        # The forced-answer turn still requested tools; return what we have.
        return text or "PerceptionAgent produced no answer."

    async def _exec_tool(self, call) -> str:
        args = dict(call.args or {})
        logger.info("PerceptionAgent: calling %s with args %r", call.name, args)
        try:
            if call.name == "frame_inspect_tool":
                return await frame_inspect_tool(
                    str(args.get("question", "")),
                    args.get("time_range"),
                    str(args.get("cue", "")),
                    ctx=self.ctx,
                )
            if call.name == "interval_summary_tool":
                return await interval_summary_tool(
                    str(args.get("question", "")),
                    args.get("time_range"),
                    ctx=self.ctx,
                )
            if call.name == "frame_associate_tool":
                cue = args.get("cue")
                return await frame_associate_tool(
                    str(args.get("question", "")),
                    list(cue) if isinstance(cue, list | tuple) else [str(cue)],
                    ctx=self.ctx,
                )
        except Exception as exc:
            logger.exception("PerceptionAgent: tool %s failed", call.name)
            return f"Error executing tool '{call.name}': {exc}"
        return f"Invalid function name: {call.name!r}"
