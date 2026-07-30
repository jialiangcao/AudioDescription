"""SubtitleAgent: one LLM pass over the video's dialogue transcript.

Port of Symphony's SubtitleAgent, sourced from the pipeline's own transcript
(Timeline segments' audio analysis) instead of subtitle JSON files. The whole
transcript goes into a single call that returns a JSON summary; the raw text
lands in the orchestrator history unparsed, as in Symphony.
"""

import logging

from qa.config import THINKING_AGENT
from qa.llm import generate_text
from qa.prompts import SUBTITLE_PROMPT, SUBTITLE_SYSTEM_PROMPT
from qa.utils import convert_seconds_to_hhmmss, with_retries
from timeline import Timeline

logger = logging.getLogger(__name__)

NO_SUBTITLES = "No subtitles available."


class SubtitleAgent:
    def __init__(self, client, question: str, timeline: Timeline):
        self.client = client
        self.question = question
        self.timeline = timeline

    def _format_transcript(self) -> str:
        entries = [
            f"{convert_seconds_to_hhmmss(seg.start)}"
            f"-{convert_seconds_to_hhmmss(seg.end)}: {seg.audio.transcript}"
            for seg in self.timeline.segments
            if seg.audio and seg.audio.has_speech and seg.audio.transcript
        ]
        return " ".join(entries)

    async def run(self) -> str:
        transcript = self._format_transcript()
        if not transcript:
            logger.info("SubtitleAgent: no dialogue transcript on the timeline")
            return NO_SUBTITLES

        prompt = SUBTITLE_PROMPT.format(question=self.question, subtitles=transcript)
        text = await with_retries(
            lambda: generate_text(
                self.client,
                system=SUBTITLE_SYSTEM_PROMPT,
                user=prompt,
                thinking=THINKING_AGENT,
            )
        )
        return text
