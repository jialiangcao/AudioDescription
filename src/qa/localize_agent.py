"""LocalizeAgent: one tool-routing LLM call -> exactly one tool execution.

Port of Symphony's LocalizeAgent: the prompt classifies the question as Type
0/1/2 and the model calls `finish` (echo a time range already in the
question), `retrieve_tool` (cheap CLIP grounding), or `localize_tool`
(exhaustive VLM scoring). Only the first tool call is executed. Unlike
Symphony, `finish` returns its answer as a plain string instead of escaping
as an exception.
"""

import logging

from google.genai import types

from qa.config import NO_TOOL_CALL_RETRIES
from qa.llm import ToolContext, generate_with_tools, user_content
from qa.prompts import LOCALIZE_AGENT_PROMPT, LOCALIZE_SYSTEM_PROMPT
from qa.tools_localize import LOCALIZE_TOOL_DECLARATION, localize_tool
from qa.tools_perception import RETRIEVE_TOOL_DECLARATION, retrieve_tool
from qa.utils import convert_seconds_to_hhmmss, with_retries

logger = logging.getLogger(__name__)

FINISH_DECLARATION = types.FunctionDeclaration(
    name="finish",
    description="For Type 0, return the complete positioning result directly.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "answer": types.Schema(
                type=types.Type.STRING,
                description=(
                    "The complete positioning result; do not directly answer "
                    "the question."
                ),
            ),
        },
        required=["answer"],
    ),
)


class LocalizeAgent:
    def __init__(
        self, client, question: str, video_duration_sec: float, ctx: ToolContext
    ):
        self.client = client
        self.ctx = ctx
        self.user_prompt = LOCALIZE_AGENT_PROMPT.replace(
            "VIDEO_LENGTH", convert_seconds_to_hhmmss(video_duration_sec)
        ).replace("QUESTION_PLACEHOLDER", question)
        self.tools = [
            types.Tool(
                function_declarations=[
                    LOCALIZE_TOOL_DECLARATION,
                    RETRIEVE_TOOL_DECLARATION,
                    FINISH_DECLARATION,
                ]
            )
        ]

    async def run(self) -> str:
        contents = [user_content(self.user_prompt)]

        response = None
        for attempt in range(NO_TOOL_CALL_RETRIES):
            response = await with_retries(
                lambda: generate_with_tools(
                    self.client,
                    system=LOCALIZE_SYSTEM_PROMPT,
                    contents=contents,
                    tools=self.tools,
                )
            )
            if response.function_calls:
                break
            logger.info(
                "LocalizeAgent: no tool call (attempt %d), retrying", attempt + 1
            )

        if response is None or not response.function_calls:
            text = (response.text if response is not None else None) or ""
            return text or "No action taken."

        # Only the first tool call is executed, as in Symphony.
        call = response.function_calls[0]
        result = await self._exec_tool(call)
        return str(result)

    async def _exec_tool(self, call) -> str:
        args = dict(call.args or {})
        logger.info("LocalizeAgent: calling %s with args %r", call.name, args)
        try:
            if call.name == "finish":
                return str(args.get("answer", ""))
            if call.name == "localize_tool":
                return await localize_tool(str(args.get("question", "")), ctx=self.ctx)
            if call.name == "retrieve_tool":
                return await retrieve_tool(str(args.get("cue", "")), ctx=self.ctx)
        except Exception as exc:
            logger.exception("LocalizeAgent: tool %s failed", call.name)
            return f"Error: {exc}"
        return f"Invalid function name: {call.name!r}"
