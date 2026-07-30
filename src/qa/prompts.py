"""Prompt set for the multi-agent Q&A system, adapted from Symphony's
lv_manager prompt set and tool prompts.

Adaptations (see qaPLAN.md): answers are free-form rather than multiple-choice
option letters; frame paths / video durations are injected by the agents so
they never appear as tool parameters; the localize windows are described as 30
seconds (matching the actual window size).

Placeholder convention: templates that contain literal JSON braces are filled
via ``str.replace`` on UPPERCASE placeholders (Symphony's own technique —
``str.format`` would trip on the braces); brace-free templates use ``.format``
or f-string builders.
"""

import json

# --------------------------------------------------------------------------- #
# CoreAgent (planner)
# --------------------------------------------------------------------------- #

CORE_SYSTEM_PROMPT = """You are a helpful assistant that answers multi-step long-form video-understanding questions by sequentially calling specialized Agents. Adhere to the THINK → ACT → OBSERVE loop for each step:
- THINK: Reason step-by-step to determine the most appropriate Agent to call next. Develop a clear plan.
- ACT: Execute your plan by calling exactly one Agent. Use arguments sourced verbatim from the user's question or previous Agent outputs—do not fabricate or infer arguments.
- OBSERVE: Summarize the output received from the Agent.
After each observation, reflect thoroughly on the results before planning the next step. Continue this loop until you have obtained all necessary information to provide a complete and accurate final answer to the user's original query.
When questions involve video content, use the available Agents to inspect it directly rather than making assumptions."""


def build_core_prompt(question: str, history: list[dict], duration: str) -> str:
    history_str = (
        "\n".join(json.dumps(item) for item in history)
        if history
        else "No actions have been taken yet."
    )
    return f"""You have three specialized agents you can delegate tasks to:
1.  `LocalizeAgent`: Use this agent to find the precise timestamp of a specific event, action, or object appearance in the video. Use this agent when the question does not provide a specific time range. This agent does not require an `instruct` field.
2.  `PerceptionAgent`: Use this agent to analyze the visual content of the video at various specific times or within multiple time ranges. It can describe scenes, identify objects, recognize actions, and answer questions about what is visually happening. This agent requires a detailed instruct field of the time segments to be analyzed.
3.  `SubtitleAgent`: Use this agent to obtain the video's dialogue transcript. This agent does not require an instruct field.

Your should:
1.  **Analyze**: Carefully examine the user's question and the execution history.
2.  **Strategize**: Decide which agent is best suited for the current need.
3.  **Conclude**: Once the execution history contains enough information to fully answer the user's question, call the `finish` agent and provide a comprehensive final answer in the `answer` field.

**Critical Rules:**
1. Carefully read the timestamps and narrations in the provided history, paying close attention to the causal sequence of events, object details and movements, and the actions and poses of people.
2. In cases where different segments lead to contradictory conclusions, use PerceptionAgent to conduct a one-time check of multiple time nodes, which involves identifying and resolving conflicts to synthesize a unique and reliable conclusion.
3. All responses must be based on the information observed by the available agents.
4. To achieve a comprehensive positioning for questions, call the LocalizeAgent with the question. Pay close attention to the information within a few minutes before and after the period when the relative score >= 3.
5. The scene descriptions provided by the LocalizeAgent should not be fully trusted. They **must** be double-checked by using the PerceptionAgent for secondary perception and confirmation.

**To call `LocalizeAgent` or `SubtitleAgent`:**
{{
    "reason": "Why this agent is the best choice for the task.",
    "agent": "AGENT_NAME",
}}
(Replace AGENT_NAME with `LocalizeAgent` or `SubtitleAgent`)

**To call `PerceptionAgent`:**
{{
    "reason": "Why this agent is the best choice for the task.",
    "agent": "PerceptionAgent",
    "instruct": "Please retrieve and return information for the following time ranges: [], [] . The elements you need to focus on include (), (), ()" # Each [] should be replaced with a specific time range (e.g., [00:06:00, 00:06:59]). Adjust the number of time ranges according to actual needs. Each () should be replaced with a phrase containing an entity, such as (number of mice), (prevent Liverpool's shot). Keep the number of phrases to three or fewer and maintain brevity in the phrases.
}}

**To finish the task and provide the final answer:**
{{
    "reason": "Why the final answer can be given now",
    "agent": "finish",
    "answer": "The final, comprehensive answer to the user's question. " # A direct, complete answer to the user's question, in one or a few sentences,
}}

The user's question is: "{question}"
Video duration: "{duration}"

Here is the execution history so far:
<history>
{history_str}
</history>

Based on the question and history, determine the next step.
Please ensure every output is in valid JSON format. Your first output character should be {{"""


# --------------------------------------------------------------------------- #
# LocalizeAgent
# --------------------------------------------------------------------------- #

LOCALIZE_SYSTEM_PROMPT = """You are a helpful assistant who answers multi-step questions by sequentially invoking functions. Follow the steps:
  • Step1: Reason step-by-step about which function to call next.
  • Step2:   Call exactly one function that moves you closer to the final answer.
  • Step3: Summarize the function's output.
"""

# Filled via .replace on QUESTION_PLACEHOLDER / VIDEO_LENGTH.
LOCALIZE_AGENT_PROMPT = """
You are an agent responsible for localizing important time points in a video that are highly relevant to the given question. The ultimate goal of the multi-agent system is to answer this question.

## Question Information

- **Question:** QUESTION_PLACEHOLDER
- **Video Duration:** VIDEO_LENGTH

## Question Analysis Process

1.  **Deep Analysis of the Question:** Understand the reasoning requirements by analyzing the question.
2.  **Identify Core Verbs/Intentions:** Determine if the core of the question is "why," "how," "cause," "result," etc., to identify causal relationships.
3.  **Extract Key Events and Entities:** Identify specific events (e.g., "taking a flower out of the bottle") and entities (e.g., "vlogger") mentioned in the question.
4.  **Infer Hidden Information:** The question may only describe an action, but the answer might require the motive or reason behind it. Therefore, the search target should not only be the scenes mentioned in the question but also any clues that might explain the motivation (e.g., dialogue, preceding and succeeding events).

## Tool Descriptions

*   **`retrieve_tool`**: Retrieves the most relevant time points from a video based on a textual cue.
    *   **Use Case:** Simple perception questions where the target is a specific object or a scene that can be described with a few keywords.
    *   **Parameters:**
        *   `cue`: A short descriptive text.
    *   **Returns:** A list of timestamps.

*   **`localize_tool`**: Designed for more complex questions that require a deeper understanding of the video content, such as identifying actions, events, or scenarios.
    *   **Use Case:** Complex questions requiring scenario understanding.
    *   **Parameters:**
        *   `question`: The question to be answered.
    *   **Returns:** A list of relevant segments, each with a timestamp, caption, relevance score (1-4, 4 is the maximum), and a justification.

*   **`finish`**: Returns the localization result.
    *   **Parameters:**
        *   `answer`: Return the complete positioning result; do not directly answer the question.

## Tool Selection based on Question Type

*   **Type 0:** The question involves a specific time range.
*   **Type 1:** The question does not involve any action, is a simple perception question, and contains detailed scene/character descriptions. The character references are clear, and there is no ambiguity in the question.
*   **Type 2:** The question is complex (requiring understanding of scenarios from the question) or is non-intuitive/abstract.

## Tool Usage Guidelines

*   **For Type 0:**
    *   For questions that involve a specific time range, directly call the `finish` tool and return that time range.

*   **For Type 1:**
    *   For questions with clear scene descriptions, no action involved, and only requiring localization of relevant time points based on scene description, directly call `retrieve_tool` for scene localization.

*   **For Type 2:**
    *   Use `localize_tool` to achieve more comprehensive and accurate positioning.
"""


# --------------------------------------------------------------------------- #
# PerceptionAgent
# --------------------------------------------------------------------------- #

PERCEPTION_SYSTEM_PROMPT = """You are a helpful assistant who answers multi-step questions by sequentially invoking functions. Follow the steps:
  • Step1: Reason step-by-step about which function to call next.
  • Step2:   Call exactly one function that moves you closer to the final answer.
  • Step3: Summarize the function's output.
"""

# Filled via .replace on Instruct_PLACEHOLDER.
PERCEPTION_AGENT_PROMPT = """You are an agent responsible for video content perception. You will receive an Instruct from an upstream agent.
**Instruct:**
<Instruct>
Instruct_PLACEHOLDER
</Instruct>

**Task:**
Follow the Instruct and use tools to analyze video content to obtain key information.

**Tool Usage Guidelines:**
*   **Video Multimodal Content Viewing:**
- To retrieve detailed information, call the frame_inspect_tool with the time range [HH:MM:SS, HH:MM:SS]. Ensure the time range is !!! longer than 5 seconds and !! less than 60 seconds!!!. If inspecting a longer duration, break it into multiple consecutive ranges of 60 seconds and prioritize checking them in order of relevance. The end time should not exceed the total duration of the video.
- If you want to obtain a rough overview / background of a long period of time (!!! entire video, or time range more than 3 minutes!!!), use the interval_summary_tool with the time (in the format [HH:MM:SS, HH:MM:SS]).
- If the question involves multiple scenes, call the frame_associate_tool with a list of scene description to get the answer.
- If need to identify the !!sequence of scenes!!, use frame_associate_tool with the description of each scene.

**Invocation Rules:**
1.  You can call the tools multiple times to complete the task specified in the Instruct. In particular, you can use the frame_inspect_tool to iteratively perceive multiple segments.
2.  Call only one tool at a time.
3.  Do not include unnecessary line breaks in the tool parameters.
4.  When providing the time_range parameter, ensure correct time unit formatting. For example, 03:21 means 3 minute and 21 seconds, which should be written as 00:03:21, not 03:21:00. Pay special attention to this.

**Task Completion:**
When the task is completed, summarize the conversation content (i.e., the completion result of the perception task) and respond to the Instruct starting with [answer], after which no further tools should be called.
"""

FORCE_ANSWER_PROMPT = (
    "Force return answer: Please summarize the conversation and answer the "
    "Instruct, starting with [answer]"
)


# --------------------------------------------------------------------------- #
# SubtitleAgent (fed from the pipeline's dialogue transcript, not subtitle files)
# --------------------------------------------------------------------------- #

SUBTITLE_SYSTEM_PROMPT = "You are a specialized Dialogue Transcript Analysis agent."

SUBTITLE_PROMPT = """
Your task is to analyze the video's dialogue transcript based on the user's question.

Based on the following information:
The original video understanding question:  {question}
The full video dialogue transcript for analysis: {subtitles}

Your Analysis Task:
1. Question-relevant Analysis: Extract transcript segments directly related to the question from the original transcript.
2. Entity and Sentiment Identification: Use the transcript information to identify key entities mentioned and their associated sentiment.
3. General Content Summary: Provide a brief, high-level summary of the overall topic covered in the transcript content.

Please respond strictly in the following JSON format:
{{
  "relevant_subtitle_info": "A multi-line string containing the most relevant transcript segments. Format each entry as:\n[HH:MM:SS - HH:MM:SS]: Actual dialogue text.\nFor example:\n[00:15:32 - 00:15:35]: ...\n[00:18:05 - 00:18:09]: ...",
  "key_entities_and_sentiment": "A brief, descriptive summary of the main entities and their sentiment.",
  "overall_topic": "A one-sentence summary of the main topic discussed in the video, based only on the transcript."
}}

Please return only the JSON object.
"""


# --------------------------------------------------------------------------- #
# ReflectionAgent
# --------------------------------------------------------------------------- #

REFLECTION_SYSTEM_PROMPT = (
    " You are a rigorous reflection agent. Your task is to critically evaluate"
    " the entire problem-solving process of the core agent."
    " You will be provided with the complete operational history and the"
    " proposed final answer."
    " Your goal is to identify any potential errors, such as omissions of"
    " information, reasoning fallacies, or fabrication of facts."
    " You must return the evaluation results in JSON format."
)

# Filled via .replace on HISTORY_PLACEHOLDER / QUESTION_PLACEHOLDER /
# PROPOSED_ANSWER_PLACEHOLDER (the history may contain braces).
REFLECTION_PROMPT = """Please evaluate the credibility of the entire problem-solving process and the proposed answer based on the following information:
Operations performed by the core agent to solve a video understanding problem include:
HISTORY_PLACEHOLDER

The original video understanding question:
Question: QUESTION_PLACEHOLDER

The final answer proposed by the core agent:
Proposed Answer: PROPOSED_ANSWER_PLACEHOLDER

Evaluation Criteria:
- If the process and answer are credible and correct, set "credible" to true.
- If any errors are found, set "credible" to false and provide a concise explanation stating what the issue is and why the proposed answer is incorrect.

Please respond strictly in the following JSON format:
{
"credible": boolean, // true means the answer is credible, false means it is not
"comment": "Your concise explanation. This should be null if credible is true"
}

Please return only the JSON object."""


# --------------------------------------------------------------------------- #
# localize_tool (per-window relevance judgement)
# --------------------------------------------------------------------------- #

JUDGEMENT_SYSTEM_PROMPT = (
    "You are a helpful assistant skilled in video understanding and question analysis."
)

# Filled via .replace on {USER_QUESTION} (the template holds literal JSON braces).
JUDGEMENT_PROMPT = """
You are given a sequence of video frames sampled from a 30-second video clip.
User Question: {USER_QUESTION}

Your task is to:
1. Analyze the relevance between the question and the visual content across the entire clip.
2. Output a global relevance score and description.

Please output your analysis in the following JSON format:

{
    "relevance_score": integer,  // Relevance score from 1 to 4
    "clip_caption": "string",     // Concise description of main people (with distinguishing features), key events, actions, and relationships. Focus on elements related to the question.
    "reasoning": "string",        // For scores 2, 3, and 4: explain reasoning; for score 1: use 'null'
}

### Instructions for clip_caption:
- Focus on elements related to the question. Describe **main people, objects, events, actions, and their relationships** that are visually confirmable.
- If there are multiple scenarios, describe them respectively. Pay attention to the sequence of events!
- Only describe what is **directly observable**. Do **not** infer, imagine, or fabricate scenes beyond the visual evidence.
- If the question is about counting (e.g. 'how many', 'count' appearing in the question), clearly identify the elements mentioned in the problem statement and count them.

### Scoring Criteria:
4 points: Key elements of the question are clearly visible, sufficient to directly answer the question.
3 points: Relevant elements from the question appear, but require integration with additional information to make a judgment.
2 points: No direct relevance exists, but the scene may have indirect relevance—such as visually similar objects, objects related to the action or behavior mentioned in the question, conceptual extensions of elements in the question, or associations established through logical inference from the question to the scene.
1 point: Completely unrelated scene.

### Reasoning Guidelines:
- Score 4: Briefly state which elements confirm the answer. Output the answer.
- Score 3: Explain what is missing or ambiguous (e.g., "action starts in Segment 3 but completion unclear", "person matches description but action not observed").
- Score 2: Explain how you decomposed or extended the question (e.g., "question asks about 'a musician', and a person holding a guitar appears").
- Score 1: Set reasoning to 'null'.

Be thorough, precise, and strictly grounded in visual evidence. Avoid temporal phrases like 'the first time'.
"""


# --------------------------------------------------------------------------- #
# Perception tools (frame-consuming VLM calls)
# --------------------------------------------------------------------------- #

VISION_TOOL_SYSTEM_PROMPT = "You are a helpful assistant to answer questions."

FRAME_INSPECT_PROMPT = """You are given a question and a sequence of video frame images relevant to the question. Your task is to reason exclusively using the visual content from these images to address the question.
\nQuestion: {question}\n
Analyze all provided frames: Carefully examine the visual details (e.g., objects, actions, context) across the entire sequence to infer the most plausible scenario.

**Critical Rules:**
1. If confident, output the answer.
2. If not confident (e.g., frames are blurry, irrelevant, incomplete, or contradictory), Briefly describe the scene, output only the information that is certain and relevant to the question.
3. The concepts in the question may not match the objects that appear in the scene. At this point, check whether any other object in the scene matches the question. Describe the closest match and explain the gap between it and the question's description.
4. If the question contains any information about subtitles, ignore them as there is no subtitle information in the frames. Just return the content of the scene in chronological order!
"""

INTERVAL_SUMMARY_PROMPT = """
You are given a question and a sequence of video frame images relevant to the question. Your task is to reason exclusively using the visual content from these images to address the question.
\nQuestion: {question}\n
Analyze all provided frames: Carefully examine the visual details (e.g., objects, actions, context) across the entire sequence to infer the most plausible scenario.
If confident, output the answer.
If not confident (e.g., frames are blurry, irrelevant, incomplete, or contradictory), Briefly describe the scene, output only the information that is certain and relevant to the question.

The concepts in the question may not match the objects that appear in the scene. At this point, check whether any other object in the scene matches the question. Describe the closest match and explain the gap between it and the question's description.
"""

FRAME_ASSOCIATE_PROMPT = """
You are given a question and a sequence of video frame images. These segments appear in chronological order, but they may not be continuous in terms of time.
Your task is:
First, determine whether the described scenario and scene in the question match. If they do not match, do not answer the question but only describe the scene.
If match, use the visual content from these images to address the question.
\nQuestion: {question}\n

Carefully examine the visual details (e.g., objects, actions, context) across the entire sequence to infer the most plausible scenario.
These video frames may not be sufficient to answer the question, so the first step is to determine whether more information is needed.
If very confident (There is direct evidence in the picture), output the answer.
If not confident (e.g., frames are irrelevant, incomplete, or contradictory), Briefly describe the scene, output only the information that is certain and relevant to the question.
Answer solely based on the observable information, rather than through reasoning or fabrication!
pay attention to the sequence in which the events occur and happen.
"""
