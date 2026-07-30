# Port Symphony's multi-agent QA system into adesc as `src/qa/`

## Context

adesc's current Q&A (`src/qa.py`, 249 lines) is a single-pass CLIP-retrieve → one Gemini call flow with a dead routing branch (`should_use_vlm` rigged always-false, iterative path `NotImplementedError`). The user wants it fully replaced by a faithful port of Symphony's multi-agent QA system (`/Users/jcao/Code/Research/temp/symphony`): a text-only CoreAgent planner loop dispatching LocalizeAgent / PerceptionAgent / SubtitleAgent over a shared history blackboard, gated by a one-shot ReflectionAgent, with VLM-backed tools (`localize_tool`, `retrieve_tool`, `frame_inspect_tool`, `interval_summary_tool`, `frame_associate_tool`).

Constraints & decisions (confirmed with user):
- **Gemini only** (google-genai SDK, already a dep) — no DashScope/Ark/OpenAI. LanguageBind is replaced by adesc's existing open_clip `ClipFrameRetriever`.
- Port **only** what's QA-relevant; use `lv_manager` (Symphony's developed prompt set) as the prompt source.
- Copy Symphony's logic/agent flow as exactly as possible; adapt only to fit adesc.
- Prompts adapted for **free-form answers** (Symphony is MCQ; drop "single option (A B C D)" wording, keep flow/rules/rubrics).
- `/ask` returns the **full trace** `{status, answer, cycles, history}` and the frontend gets a **trace panel** showing each cycle.
- QA becomes a **`src/qa/` package** (user explicitly wants a multi-file subsection; this is a deliberate exception to the flat-module rule — the package name keeps `server.py`'s `import qa` + `qa.answer_question` attribute access working, which the test monkeypatch idiom relies on: `tests/test_server.py`).

Post-pull data model (upstream merge `ce81b23`, already in the tree): `Segment.keyframe` / `keyframes` / `visual` no longer exist. Each segment now carries `frames: list[Frame]` where `Frame{index, time, path, visual: FrameAnalysis | None}` — `time` is the frame's **absolute timestamp** in the video, and every frame has its own per-frame analysis (`FrameAnalysis` = old `VisualAnalysis` + `actions: list[str]`). The QA port reads `frame.time`/`frame.path`; the per-frame `visual` analyses are deliberately **not** fed to the agents (Symphony has no analog — its planner only sees tool outputs), though they're available for a future enrichment. `Timeline` also gained `described_video` (mux stage) — irrelevant to QA.

## Architecture

```
server.py /ask ──► qa.answer_question(timeline, question, client)   [async]
                        │
                   qa/orchestrator.py  QASystem.run()               ← port of video_understanding.py
                        │  while cycle < MAX_CYCLES(17):
                        │    CoreAgent (Gemini, response_schema=CoreDecision)
                        │    dispatch: LocalizeAgent | PerceptionAgent | SubtitleAgent | finish→ReflectionAgent (one-shot latch)
                        │    history.append({action, reason, instruct?, result})
                        ▼
      tools (async, Gemini vision + CLIP)  ── FrameIndex (timestamps from Timeline, replaces fps math)
```

## Files

### New package `src/qa/` (delete `src/qa.py` — a package and module can't both be `qa`)

| File | Responsibility |
|---|---|
| `__init__.py` | `from qa.orchestrator import QAResult, answer_question` only |
| `config.py` | All constants (table below) |
| `utils.py` | `convert_seconds_to_hhmmss` / `convert_hhmmss_to_seconds` (port Symphony's, incl. `MM:SS` tolerance); `async fix_and_parse_json(text, client)` (strip fences → `json.loads` → one Gemini repair call → `None`); `async with_retries(fn)` (3 attempts, exp backoff — replaces Symphony's 5×60s) |
| `frame_index.py` | `FrameIndex` (below) |
| `retriever.py` | `ClipFrameRetriever` moved verbatim from old `qa.py` (`_pick_device`, lazy `_load_retriever()`, `_RETRIEVER_LOCK`) + path-keyed embedding cache + `async retrieve_top_k(paths, cue, k)` via `asyncio.to_thread` |
| `prompts.py` | All prompts adapted from `lv_manager` + Symphony tool prompts (mapping below) |
| `llm.py` | Thin async Gemini wrappers reusing `vision_analysis.py` patterns: `generate_text(client, system, user, schema=None, thinking, max_output_tokens)`, `generate_vision(client, system, user, frame_paths, schema=None)` (`types.Part.from_bytes`, `response.text is None → RuntimeError` guard), `generate_with_tools(client, system, contents, tools, thinking)`. All `temperature=0.0` |
| `tools_localize.py` | `localize_tool` + `Judgement` schema + its `FunctionDeclaration` |
| `tools_perception.py` | `retrieve_tool`, `frame_inspect_tool`, `interval_summary_tool`, `frame_associate_tool` + declarations |
| `core_agent.py` | `CoreAgent` + `CoreDecision(BaseModel){reason, agent, instruct?, answer?}` |
| `localize_agent.py` | `LocalizeAgent` (tools: localize, retrieve, finish) |
| `perception_agent.py` | `PerceptionAgent` (ReAct loop ≤6) |
| `subtitle_agent.py` | `SubtitleAgent` (Timeline-transcript-backed) |
| `reflection_agent.py` | `ReflectionAgent` + `ReflectionAssessment{credible, comment}` |
| `orchestrator.py` | `QASystem` (port of `VideoUnderstandingSystem`), `QAResult`, `answer_question` |

Internal imports package-absolute (`from qa.config import ...`).

### Modified
- `src/server.py` — `/ask` endpoint + `AskResponse` model
- `tests/conftest.py`, `tests/test_server.py`
- `frontend/app/lib/api.ts`, `frontend/app/page.tsx`, `frontend/app/globals.css`
- `CLAUDE.md` (flat-module rule exception + QA section rewrite)

## Key components

### FrameIndex (replaces Symphony's `frame_path` dir + fps=2 arithmetic)
Frozen dataclass built from `Timeline`: sorted `entries: list[(timestamp_sec, abs_path)]` = `[(frame.time, frame.path) for seg in timeline.segments for frame in seg.frames]` — `Frame.time` is authoritative (absolute, set by segmentation), so no timestamp derivation is needed; `duration_sec` from the timeline. Methods: `paths()`, `timestamp_of(path)`, `in_range(start, end)`, `windows(30.0)` (fixed grid, bucket by timestamp, drop empty windows — replaces `group_frames`), `uniform_sample(start, end, count)` (linspace `endpoint=False`, nearest frame, dedup, sorted). Note: adesc's 0.5 fps sampling equals exactly what Symphony's `localize_tool` sent per window (15 frames / 30 s), so no sub-sampling needed.

### Gemini function calling (manual loop, mirrors Symphony)
- Hand-written `types.FunctionDeclaration` per tool with real one-line descriptions (Symphony left them empty). Model-visible params only — `frame_path` / `video_duration` removed from schemas entirely. `time_range` = `Schema(ARRAY of STRING, "exactly two items, HH:MM:SS")` (Gemini doesn't support `prefixItems`).
- `config=GenerateContentConfig(system_instruction, tools=[types.Tool(function_declarations=[...])], temperature=0.0, thinking_config, max_output_tokens)`. No callables passed → automatic function calling never triggers.
- Read `response.function_calls`; args via `dict(fc.args)` (already structured — Symphony's per-arg `fix_and_parse_json` unnecessary); validate `time_range` in the tool, return error **string** on bad input (Symphony behavior).
- Context injection: replace `co_varnames` inspection with an explicit frozen `ToolContext(client, frame_index)`; agents call `await tool(**fc.args, ctx=self.ctx)`.
- Tool turn: append `resp.candidates[0].content`, then `types.Content(role="tool", parts=[types.Part.from_function_response(name=..., response={"result": str(result)}) ...])`.

### Orchestrator loop (faithful port, `video_understanding.py:86-186`)
`while not completed and cycle < MAX_CYCLES`: CoreAgent → decision dict (or `None` → append `{"action": "parse_failure", ...}` and continue — **fix**, Symphony AttributeErrors). Dispatch verbatim: `PerceptionAgent(instruct, question, duration)` / `SubtitleAgent()` / `LocalizeAgent()` / `finish` / else `{"action": "unknown_agent", "decision": ...}`. History record shapes verbatim: `{action, reason, result}` + `instruct` iff present.

Reflection gate verbatim: append finish record first; `if not if_reflected:` run ReflectionAgent, latch; else `{"credible": True}` (auto-accept). Credible → append reflection-credible + finish records, break. Not credible → append the long instructional sentence as the `"action"` value + `{"assessment": "not_credible", "comment": ...}`, continue.

Return `QAResult{status: "completed"|"failed", answer, reason, cycles, history}`.

### Agents
- **CoreAgent**: no tools; `response_schema=CoreDecision` (Gemini-native structured output) while keeping the prompt's JSON templates + "first output character should be `{`" verbatim; `fix_and_parse_json` as backstop.
- **LocalizeAgent**: retry ≤6 until `function_calls` present; execute **first call only** (verbatim); `finish` returns `str(args["answer"])` directly (**fix**: no `StopException` escaping `run()`); no-tool-call fallback returns `resp.text or "No action taken."`.
- **PerceptionAgent**: loop ≤6; last iteration appends `FORCE_ANSWER_PROMPT` (verbatim text); inner retry ≤6 when neither `[answer]` nor tool calls; `text = resp.text or ""` (**fix**: None guard); `[answer]` in text → return text; no tool calls → return text; else execute **all** tool calls, append model content + tool responses. Final fallback returns best-effort string, never `None` (**fix**).
- **SubtitleAgent** (adapted source): entries from `timeline.segments` where `seg.audio and seg.audio.has_speech and seg.audio.transcript`, formatted `HH:MM:SS-HH:MM:SS: text` joined with spaces; empty → `"No subtitles available."` without a model call; one Gemini call → raw JSON text into history (verbatim behavior, 3 keys `relevant_subtitle_info` / `key_entities_and_sentiment` / `overall_topic`).
- **ReflectionAgent**: one call, `schema=ReflectionAssessment`; any failure → `{"credible": True, "comment": "Fallback: ..."}` (**kept: fail-open**).

### Tools
- `localize_tool(question, *, ctx)`: `windows(30.0)` → one `generate_vision` per window with `JUDGEMENT_PROMPT` (schema `Judgement{relevance_score: int, clip_caption, reasoning?}`), `asyncio.gather` under `Semaphore(LOCALIZE_CONCURRENCY)`; per-window failure → logged + skipped (Symphony behavior); keep `score > 1`; window label `hhmmss(start)-hhmmss(start+29)`; return `'The relevance segment:' + str([...])`.
- `retrieve_tool(cue, *, ctx)`: CLIP top-15 over all frames → timestamps via `timestamp_of`, **similarity order** (verbatim) → `'The most similar time point:' + str([...])`.
- `frame_inspect_tool(question, time_range, cue, *, ctx)`: clamp end to duration; frames = `uniform_sample(start, end, min(70, int(end-start)))` ∪ CLIP top-20 over in-range paths, dedup, chronological; `FRAME_INSPECT_PROMPT`.
- `interval_summary_tool(question, time_range, *, ctx)`: `uniform_sample(start, end, 30)`; `INTERVAL_SUMMARY_PROMPT`.
- `frame_associate_tool(question, cue: list[str], *, ctx)`: union of CLIP top-10 per cue, chronological; `FRAME_ASSOCIATE_PROMPT`.
- Empty frame selections return an explanatory error string instead of calling the model. Vision calls wrapped in `with_retries`.

### Retriever cache
`_EMBED_CACHE: dict[path, tensor row]` (bound `MAX_CACHED_EMBEDDINGS = 20_000`, FIFO evict); under `_RETRIEVER_LOCK` encode only uncached paths, assemble matrix in caller order, then `similarity_top_k`. First tool call per job pays the embed cost; the rest are text-encode-only (today every question re-embeds every frame).

## Prompt mapping (`qa/prompts.py`)

Fill placeholders via `.replace()` when the template contains literal JSON braces (Symphony's own technique); `.format()`/f-string builders otherwise — note this convention at module top like adesc `prompts.py`.

| New constant | Symphony source | Adaptation |
|---|---|---|
| `CORE_SYSTEM_PROMPT` | `lv_manager.build_system_prompt()` | verbatim (THINK→ACT→OBSERVE) |
| `build_core_prompt(q, history, duration)` | `lv_manager.build_core_prompt` | Rule 4 "question and options"→"question"; finish comment `# single option (A B C D)` → `# A direct, complete answer to the user's question`. Keep agent roster, 5 Critical Rules (score≥3 focus; "LocalizeAgent captions must be double-checked by PerceptionAgent"), structured instruct template (`time ranges: [], []` + `≤3` focus elements), `<history>` newline-json.dumps, "first output character should be {" |
| `LOCALIZE_SYSTEM_PROMPT` / `LOCALIZE_AGENT_PROMPT` | `S_prompt_localizeagent` / `prompt_localizeagent` | system verbatim; drop options language, drop `frame_path` param lines and the phantom `localize_instruction` param (**fix**); keep 4-step Question Analysis (incl. infer-hidden-motive), Type 0/1/2 routing |
| `PERCEPTION_SYSTEM_PROMPT` / `PERCEPTION_AGENT_PROMPT` | `S_prompt_perceptionagent` / `prompt_without_q_perceptionagent` | verbatim incl. 5–60s inspect rule, >3min summary rule, `00:03:21` format warning, `[answer]` protocol; "question and options"→"question" |
| `FORCE_ANSWER_PROMPT` | inline in `A_PerceptionAgent.run` | verbatim |
| `SUBTITLE_SYSTEM_PROMPT` / `SUBTITLE_PROMPT` | `A_SubtitleAgent` inline | "subtitles"→"dialogue transcript" in framing; keep 3 JSON output keys verbatim |
| `REFLECTION_SYSTEM_PROMPT` / `REFLECTION_PROMPT` | `A_ReflectionAgent` inline | verbatim (no MCQ language present) |
| `JUDGEMENT_SYSTEM_PROMPT` / `JUDGEMENT_PROMPT` | `localize_tools.py:60` | "1-minute clip"→"30-second clip" (**fix**, matches `fenzu_time=30`); "question (including all options)"→"question"; keep 1–4 rubric, counting rule, per-score reasoning guidelines, "avoid temporal phrases" |
| `FRAME_INSPECT_PROMPT` / `INTERVAL_SUMMARY_PROMPT` / `FRAME_ASSOCIATE_PROMPT` | `perception_tools.py` inline prompts | rewrite the options-matching clause for free-form ("describe the closest match and explain the gap"); rest verbatim incl. confidence/never-fabricate rules and chronological-sequence note |
| `VISION_TOOL_SYSTEM_PROMPT` | "You are a helpful assistant to answer questions." | verbatim |
| `JSON_REPAIR_PROMPT` | `utils.fix_and_parse_json` | verbatim |

Not ported: commented-out prompt variants, `retrieve_and_ans_tool`, `associate`, `subtitle_tool`, per-benchmark managers, benchmark scripts, LanguageBind.

## Config constants (`qa/config.py`)

| Constant | Value | Note |
|---|---|---|
| `TEXT_MODEL` / `VISION_MODEL` | `"gemini-3.5-flash"` | matches `vision_analysis.MODEL`; fixes old 2.5/3.5 drift; separate constants so they can diverge |
| `THINKING_PLANNER` | `ThinkingLevel.HIGH` | CoreAgent + Reflection = Symphony's DeepSeek-R1 seats |
| `THINKING_AGENT` / `THINKING_VISION` | `ThinkingLevel.LOW` | routing/extraction; adesc found HIGH over-condenses perceptual tasks |
| `TEXT_MAX_OUTPUT_TOKENS` / `VISION_MAX_OUTPUT_TOKENS` | 4096 / 1024 | thinking shares output budget (vision: Symphony's 512 doubled) |
| `MAX_CYCLES` | 17 | Symphony |
| `PERCEPTION_MAX_ITERATIONS` / `NO_TOOL_CALL_RETRIES` | 6 / 6 | Symphony |
| `LOCALIZE_WINDOW_SEC` / `LOCALIZE_MIN_SCORE` / `LOCALIZE_CONCURRENCY` | 30.0 / keep score>1 / 8 | concurrency down from ThreadPoolExecutor(20) for Gemini limits |
| `RETRIEVE_TOP_K` / `INSPECT_UNIFORM_MAX` / `INSPECT_RETRIEVE_TOP_K` / `SUMMARY_FRAME_COUNT` / `ASSOCIATE_TOP_K_PER_CUE` | 15 / 70 / 20 / 30 / 10 | Symphony values |
| `RETRY_ATTEMPTS` / `RETRY_BASE_DELAY_SEC` | 3 / 2.0 | replaces 5×60s backoff; 120s timeout comes from injected client |
| `MAX_CACHED_EMBEDDINGS` | 20_000 | ~40 MB bound |

## Symphony bugs fixed vs. behaviors kept

**Fixed**: `StopException` escaping `LocalizeAgent.run()`; parse-`None` → AttributeError in orchestrator; PerceptionAgent `None`-content crash / implicit `None` return; JUDGEMENT "1-minute" vs 30s window; phantom `localize_instruction` param; model-name drift + per-question file logging (→ shared injected client, module loggers).

**Kept verbatim**: `if_reflected` one-shot latch; reflection fail-open; first-tool-call-only in LocalizeAgent; retry-6 loops; unbounded history re-serialized every cycle; the not-credible sentence as `"action"`; all record shapes; score>1 filter, top-k values, window math, uniform∪retrieved union; `max_cycles=17`; temperature 0.

## Server & frontend

`server.py`:
- `AskResponse{answer: str | None, status: str, cycles: int, history: list[dict]}`.
- `/ask`: keep 404/410/409 guards and 500 wrapper; body becomes `result = await qa.answer_question(job.timeline, body.question, _get_pipeline_client(app))` (direct await — QA is now async; CLIP inside uses `to_thread`); `return AskResponse(**result.model_dump())`. Keep `import qa` attribute-access style.

`frontend/app/lib/api.ts`: `TraceEntry{action, reason?, instruct?, result?, answer?, assessment?, comment?, [k: string]: unknown}`, `AskResult{answer, status, cycles, history}`; `askQuestion` returns the full object.

`frontend/app/page.tsx`: `answer` state → `AskResult | null`; render answer text (fallback when `answer` null / status failed) + `<details className="trace">` panel: summary "Reasoning trace — N cycles (status)", one card per history entry (bold action, reason, instruct if present, result in a collapsed `<pre>` with CSS max-height). `globals.css`: `.trace`, `.trace-entry` styles following existing `.qa`/`.answer` look.

## Tests

`tests/conftest.py` (backward-compatible): `FakeGeminiResponse` gains `function_calls=None` + `candidates` (SimpleNamespace with `.content`); `FakeGeminiClient.queue(*responses)` scripted FIFO; `_canned_response` becomes schema-aware (dispatch on `config.response_schema`: FrameAnalysis → existing canned JSON, CoreDecision → finish decision, ReflectionAssessment → credible, Judgement → score-1; else canned sentence). New `fake_frame_index` and `stub_retriever` (monkeypatch `qa.retriever.retrieve_top_k`) fixtures.

New files (asyncio_mode=auto, prompt assertions via substring idiom from `test_vision_analysis.py`):
- `test_qa_frame_index.py` — entries taken from `Frame.time`/`Frame.path` in order, `in_range` bounds, `windows(30)` bucketing/empty-drop, `uniform_sample`.
- `test_qa_orchestrator.py` — dispatch + record shapes, finish→credible 3-record tail, not-credible→continue, **latch** (2nd finish skips reflection), max-cycles→failed, core-`None`→parse_failure, unknown agent.
- `test_qa_agents.py` — Core prompt substrings + parse + backstop; Localize finish→string / first-call-only / retry fallback; Perception scripted tool→response→`[answer]`, force-answer on last iteration, None guards; Subtitle formatting + no-speech short-circuit (no model call); Reflection fail-open.
- `test_qa_tools.py` — localize fan-out count + score filter + output format; retrieve format; inspect union/dedup/chronological/clamp; summary count; associate per-cue union.
- `test_qa_retriever.py` — cache: second retrieve over same paths encodes 0 new images.
- `test_server.py` — ask stub becomes `async def` returning a `QAResult`-shaped object; assert new response fields; keep `monkeypatch.setattr(qa, "answer_question", ...)`.

## Step ordering (each step leaves `uv run pytest` green)

1. **Scaffold + retriever move**: create `src/qa/` (`__init__.py` with placeholder `answer_question` so `import qa` + monkeypatch keep working), `config.py`, `utils.py`, `frame_index.py`, `retriever.py`; **delete `src/qa.py`**. Tests: frame_index, retriever cache, utils.
2. **Prompts**: `qa/prompts.py` + prompt-content tests.
3. **LLM layer + fakes**: `qa/llm.py`; extend conftest.
4. **Tools**: `tools_localize.py`, `tools_perception.py` + `test_qa_tools.py`.
5. **Agents + orchestrator**: five agent modules, `orchestrator.py`, real `__init__` re-export + agent/orchestrator tests.
6. **Server**: `/ask`, `AskResponse`; update `test_server.py`.
7. **Frontend**: `api.ts`, `page.tsx`, `globals.css`.
8. **Docs**: CLAUDE.md (flat-rule exception, QA section), README blurb if stale.

## Verification

```bash
cd /Users/jcao/Code/Research/temp/worktree-qa
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run pyright src
# Manual e2e (needs GEMINI_API_KEY in .env, ffmpeg/espeak-ng on PATH):
./run.sh   # backend :8000 + frontend :3000
# Upload a short clip; once done:
#  (a) simple perceptual question → expect Localize→Perception→finish in trace
#  (b) dialogue question → expect SubtitleAgent in trace
#  (c) trace panel shows per-cycle agent/reason/result; repeat question is faster (embed cache)
# ADESC_LOG_LEVEL=DEBUG for cycle-by-cycle backend logs.
```
