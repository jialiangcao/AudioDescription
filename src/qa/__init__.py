"""Multi-agent video Q&A (a port of Symphony's agent system onto Gemini).

This package is a deliberate exception to the repo's flat-module rule: it
keeps the old ``import qa`` / ``qa.answer_question`` surface while housing the
orchestrator, agents, tools, prompts, and retriever in separate modules.
"""

from qa.orchestrator import QAResult, answer_question

__all__ = ["QAResult", "answer_question"]
