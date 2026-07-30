"""Tuning constants for the multi-agent Q&A system.

Values marked "Symphony" are carried over unchanged from the Symphony
implementation this package ports; the rest adapt it to Gemini and to this
codebase (see qaPLAN.md / CLAUDE.md).
"""

from google.genai import types

# One model for the text (planner/agent) layer and one for the frame-consuming
# vision calls. Both are Gemini per project policy; kept as separate constants
# so they can diverge later. Matches vision_analysis.MODEL.
TEXT_MODEL = "gemini-3.5-flash"
VISION_MODEL = "gemini-3.5-flash"

# Thinking effort per seat. CoreAgent/ReflectionAgent fill Symphony's
# DeepSeek-R1 reasoning seats, so they get HIGH; the worker agents are
# routing/extraction and the vision tools are perceptual — HIGH tends to
# over-condense those (see vision_analysis.py), so they stay LOW.
THINKING_PLANNER = types.ThinkingLevel.HIGH
THINKING_AGENT = types.ThinkingLevel.LOW
THINKING_VISION = types.ThinkingLevel.LOW

# Reasoning tokens share the output budget on Gemini, so the planner budget is
# generous; vision calls answer in prose (Symphony capped them at 512 visible
# tokens — doubled here to leave room for LOW thinking).
TEXT_MAX_OUTPUT_TOKENS = 4096
VISION_MAX_OUTPUT_TOKENS = 1024

# Orchestrator cycle budget. (Symphony)
MAX_CYCLES = 17

# PerceptionAgent ReAct iterations; the last one force-requests an answer. (Symphony)
PERCEPTION_MAX_ITERATIONS = 6

# How often Localize/Perception re-sample the model when it returns neither an
# answer nor a tool call. (Symphony)
NO_TOOL_CALL_RETRIES = 6

# localize_tool scores the whole video in fixed windows of this many seconds.
# (Symphony fenzu_time)
LOCALIZE_WINDOW_SEC = 30.0

# Windows with relevance_score below this are dropped (Symphony kept score > 1).
LOCALIZE_MIN_SCORE = 2

# Concurrent Gemini vision calls inside localize_tool. Symphony used a
# ThreadPoolExecutor(20); kept lower here for Gemini rate limits.
LOCALIZE_CONCURRENCY = 8

# Frame-selection sizes, all Symphony values.
RETRIEVE_TOP_K = 15
INSPECT_UNIFORM_MAX = 70
INSPECT_RETRIEVE_TOP_K = 20
SUMMARY_FRAME_COUNT = 30
ASSOCIATE_TOP_K_PER_CUE = 10

# Transient-error retry for Gemini calls. Replaces Symphony's 5×(60s, doubling)
# backoff; the 120s per-request timeout comes from the injected client.
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY_SEC = 2.0

# Bound on the CLIP frame-embedding cache (rows are ~512 floats ≈ 2KB, so this
# caps the cache near 40MB on a long-lived server).
MAX_CACHED_EMBEDDINGS = 20_000
