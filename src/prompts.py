"""Prompt templates for audio-description generation.

Transcribed from the GenAD prompt templates (the paper's Appendix A: system
prompt with AD guidelines, scene-level generation, and inline optimization).
The templates are used verbatim except for the JSON/output-format instructions,
which are adapted to the structures this pipeline consumes:

* the scene-level generation prompt emits a single JSON object matching
  ``vision_analysis.ShotAnalysis`` (description/entities/setting/on_screen_text)
  instead of the paper's ``start_time``/``type``/``text`` event array;
* everything else — the guidelines, the system message, the character-naming
  rules, and the entire inline-optimization prompt — is left as written.
"""

# --- System prompt (Appendix A.1) -------------------------------------------

AD_GUIDELINES = """AUDIO DESCRIPTION GUIDELINES:
- Describe what you see in a concise, factual manner.
- Always read on-screen text exactly as it appears.
- Be factual, objective, and precise in your descriptions.
- Use proper terminology and names from the context when possible.
- Match the tone and mood of the video.
- Do not over-describe -- less is more.
- Do not interpret or editorialize about what you see.
- Do not give away surprises before they happen.

CHARACTER IDENTIFICATION:
- When you recognize a character from the context, ALWAYS use their specific name.
- Before each scene, carefully review context to identify all named characters.
- Use the most specific identification possible based on the context information."""

_SYSTEM_MESSAGE = """You are an expert audio describer creating descriptions for videos \
to enhance accessibility for blind and low-vision users. Your task is to generate video \
descriptions that are true, contextually meaningful, useful, well-timed, and polished. \
You must follow the audio description guidelines below.

{guidelines}"""

# The composed system instruction, guidelines injected via the {guidelines} placeholder.
SYSTEM_INSTRUCTION = _SYSTEM_MESSAGE.format(guidelines=AD_GUIDELINES)


# --- Scene-level generation prompt (Appendix A.2) ---------------------------
# Verbatim except the JSON/output instructions, adapted to emit one ShotAnalysis
# object rather than the paper's array of timed Text-on-Screen / Visual events.

SCENE_GENERATION_PROMPT = """SCENE DURATION: {scene_duration:.2f} seconds

CONTEXT:
{context}

You are analyzing a video scene. Identify specific characters, locations, and any \
important elements mentioned in the context.

First, capture Text on Screen:
- Capture ALL visible on-screen text.
- DO NOT include transcript or dialogue.

INCLUDE: Titles, headings, names; informational text; important dates or events.
EXCLUDE: Brand logos and watermarks; network logos; social media handles; copyright notices.

Second, generate the visual description:
- Provide contextually rich visual description of the scene using concise wording.
- Describe each action in detail.
- ALWAYS use specific character names from context (not "person" or "woman").
- Focus on key actions, settings, and objects not mentioned in a previous description.
- DO NOT describe Text on Screen.

OUTPUT FORMAT (JSON object):
  - description     (the visual description text)
  - entities        (array of the specific characters and objects present)
  - setting         (the location or setting of the scene)
  - on_screen_text  (all captured on-screen text, or null if there is none)"""

# Default context when no prior narrative context is available (per-keyframe
# analysis runs concurrently, so there is no accumulated scene history to feed).
NO_CONTEXT = (
    "(No prior narrative context is available; identify the characters, location, "
    "and key elements directly from the frame.)"
)


# --- Narration line prompt --------------------------------------------------
# The per-gap audio-description line written for a blind/low-vision (BLV) viewer.
# Split into a header (dynamic: gap length + word budget) and a static guidelines
# block so the current-shot and earlier-scene context can be concatenated in
# safely (their text may contain braces, which str.format would choke on). The
# earlier-scene context gives the model the prior descriptions, dialogue, and
# narration so it maintains continuity: reusing established names, not repeating
# visuals, and building on what came before.

NARRATION_HEADER = """You are writing a single line of audio description (AD) narration for a \
blind or low-vision (BLV) viewer. It will be spoken during a {gap_sec:.1f}-second speech-free \
gap in the current shot, so it must fit that time — about {max_words} words or fewer. The line \
complements the video: it conveys essential visual information the viewer cannot see, without \
restating the dialogue or narrating the filmmaking."""

NARRATION_GUIDELINES = """GUIDELINES:
- Describe only what is visible now, in the present tense, concisely and factually.
- Lead with the most important information: who, what, where, and the key action.
- Reuse the names and places established in earlier scenes; never re-introduce someone already \
named (e.g. do not call a named character "a man" or "a woman" again).
- Add only what is new — do not repeat visual details already given in earlier scenes, or facts \
already clear from the dialogue.
- Describe actions and setting directly. Never write "we see", "the camera shows", "in this \
scene", or otherwise narrate the production.
- Do not interpret motives, assign emotions, or reveal surprises before they happen.
- Keep the language natural and vivid but economical, matching the tone of the video.

GOOD EXAMPLES:
- "Maria tucks the letter into her coat and hurries down the rain-slick alley."
- "The workshop is empty now, tools hanging in neat rows above the bench."
- "David freezes, his hand hovering over the door handle."

BAD EXAMPLES:
- "We see a woman walking away." — uses "we see", and she was already named Maria.
- "A man looks sad and probably regrets what he did." — guesses feelings and motives.
- "Maria is in a room with a table; the lighting is warm and cinematic." — disconnected \
details that also comment on the filmmaking.

Return only the narration line — no quotation marks, labels, or explanation."""

# Header for the earlier-scene continuity block, and the fallback when the shot
# is the first one described.
NARRATION_PRIOR_HEADER = (
    "EARLIER SCENES (oldest first) — maintain continuity with these and do not repeat their "
    "information:"
)
NARRATION_NO_PRIOR = "EARLIER SCENES: none — this is the first described shot."


# --- Inline optimization prompt (Appendix A.3) ------------------------------
# Used verbatim (no JSON to adapt): condenses a too-long description so its
# spoken duration fits the available silence gap.

INLINE_OPTIMIZATION_PROMPT = """You are optimizing a set of visual descriptions for a video.

ORIGINAL DESCRIPTIONS: "{combined_text}"
AVAILABLE TIME: {available_duration:.2f} seconds

TASK:
Combine and condense these descriptions to fit within {available_duration:.2f} seconds \
of spoken audio.

GUIDELINES:
- Create a coherent, flowing description.
- Maintain the action order and use concise but natural language.
- Ensure the final description can be spoken within the time limit.

OUTPUT: Provide only the optimized description text, without explanations."""


# --- Retry optimization prompt (Appendix A.4) -------------------------------
# Used verbatim. Fired when the inline-optimized line, once synthesized, still
# overruns its gap (verified via TTS): it reports the measured speech duration
# and the overshoot so the model can cut the right amount. ``{reduce_by}`` holds
# the precomputed ``tts_duration - available_duration`` (str.format can't do the
# subtraction inline), so the rendered text reads exactly as in the paper.

RETRY_OPTIMIZATION_PROMPT = """You are optimizing visual descriptions for a video.

PREVIOUS ATTEMPT: "{optimized_text}"

This description takes {tts_duration:.2f} seconds to speak, but only \
{available_duration:.2f} seconds are available. Reduce by {reduce_by:.2f} seconds.

TASK:
Create a SHORTER version that fits within the time limit.

GUIDELINES:
- Keep the most critical visual elements.
- Eliminate redundant details.
- Use concise but natural language.

OUTPUT: Provide only the shortened description, nothing else."""
