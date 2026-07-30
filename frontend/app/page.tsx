"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  askQuestion,
  describedVideoUrl,
  frameUrl,
  narrationUrl,
  uploadVideo,
  wsBase,
  type AskResult,
  type Frame,
  type JobStatus,
  type PipelineEvent,
  type Segment,
  type TraceEntry,
} from "./lib/api";

const STAGE_ORDER = [
  "segmentation",
  "vision",
  "audio",
  "timeline",
  "narration",
  "tts",
  "mux",
];
const STAGE_LABELS: Record<string, string> = {
  segmentation: "Shots",
  vision: "Frames",
  audio: "Audio",
  timeline: "Timeline",
  narration: "Narration",
  tts: "Voice",
  mux: "Mix",
};

/** Absolute video timestamp as m:ss.s. */
function fmtTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

/**
 * Merge incoming frames into a shot's existing frame list by `index`.
 *
 * Frame analyses arrive out of order and after the shot skeleton, so neither
 * side can be treated as authoritative — each frame is merged field-by-field
 * and the result is re-sorted into temporal order.
 */
function mergeFrames(
  existing: Frame[] | undefined,
  incoming: Partial<Frame>[],
): Frame[] {
  const byIndex = new Map<number, Partial<Frame>>();
  for (const frame of existing ?? []) byIndex.set(frame.index, frame);
  for (const frame of incoming) {
    if (frame.index == null) continue;
    byIndex.set(frame.index, { ...byIndex.get(frame.index), ...frame });
  }
  return [...byIndex.values()]
    .sort((a, b) => (a.index ?? 0) - (b.index ?? 0))
    .map((frame) => ({
      index: frame.index ?? 0,
      time: frame.time ?? 0,
      path: frame.path ?? "",
      visual: frame.visual ?? null,
    }));
}

export default function Home() {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [stages, setStages] = useState<Record<string, "active" | "done">>({});
  const [segments, setSegments] = useState<Record<number, Partial<Segment>>>(
    {},
  );
  const [described, setDescribed] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<AskResult | null>(null);
  const [asking, setAsking] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const startJob = useCallback(async (file: File) => {
    setError(null);
    setAnswer(null);
    setSegments({});
    setStages({});
    setDescribed(false);
    setStatus(null);
    setVideoUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(file);
    });
    try {
      const id = await uploadVideo(file);
      setJobId(id);
      setStatus("queued");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  // Stream live pipeline events over the websocket while a job is active.
  useEffect(() => {
    if (!jobId) return;
    const ws = new WebSocket(`${wsBase()}/api/jobs/${jobId}/events`);

    ws.onmessage = (msg) => {
      const event: PipelineEvent = JSON.parse(msg.data);
      switch (event.type) {
        case "stage":
          setStages((prev) => ({
            ...prev,
            [event.stage]: event.status === "done" ? "done" : "active",
          }));
          break;
        case "shots":
          // Skeleton: every extracted frame with its timestamp, before any of
          // them has been described.
          setSegments((prev) => {
            const next = { ...prev };
            for (const shot of event.shots) {
              next[shot.id] = {
                ...next[shot.id],
                id: shot.id,
                start: shot.start,
                end: shot.end,
                frames: mergeFrames(next[shot.id]?.frames, shot.frames),
              };
            }
            return next;
          });
          break;
        case "frame":
          setSegments((prev) => ({
            ...prev,
            [event.shot_id]: {
              ...prev[event.shot_id],
              id: event.shot_id,
              frames: mergeFrames(prev[event.shot_id]?.frames, [
                {
                  index: event.index,
                  time: event.time,
                  path: event.path,
                  visual: event.visual,
                },
              ]),
            },
          }));
          break;
        case "timeline":
          setSegments((prev) => {
            const next = { ...prev };
            for (const seg of event.timeline.segments) {
              next[seg.id] = {
                ...next[seg.id],
                ...seg,
                frames: mergeFrames(next[seg.id]?.frames, seg.frames),
              };
            }
            return next;
          });
          break;
        case "narration":
          setSegments((prev) => ({
            ...prev,
            [event.segment_id]: {
              ...prev[event.segment_id],
              ad_narration: event.text,
            },
          }));
          break;
        case "narration_audio":
          setSegments((prev) => ({
            ...prev,
            [event.segment_id]: {
              ...prev[event.segment_id],
              ad_narration_audio: event.audio,
              ad_narration_duration_sec: event.duration_sec,
              ad_narration_overflow: event.overflow,
            },
          }));
          break;
        case "described_video":
          setDescribed(true);
          break;
        case "status":
          setStatus(event.status);
          if (event.status === "error") {
            setError(event.error ?? "processing failed");
          }
          break;
      }
    };
    ws.onerror = () => setError("lost connection to server");

    return () => ws.close();
  }, [jobId]);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) startJob(file);
  };

  const onAsk = async () => {
    if (!jobId || !question.trim()) return;
    setAsking(true);
    setAnswer(null);
    setError(null);
    try {
      setAnswer(await askQuestion(jobId, question.trim()));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setAsking(false);
    }
  };

  const orderedSegments = Object.values(segments).sort(
    (a, b) => (a.id ?? 0) - (b.id ?? 0),
  );
  const allFrames = orderedSegments.flatMap((seg) => seg.frames ?? []);
  const describedCount = allFrames.filter((f) => f.visual).length;
  const done = status === "done";

  return (
    <main className="container">
      <h1>Audio Description</h1>
      <p className="subtitle">
        Drop a video to generate audio-description narration, then ask questions
        about it.
      </p>

      <div
        className={`dropzone${dragging ? " drag" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        {jobId
          ? "Drop another video to start over"
          : "Drag & drop a video here, or click to choose a file"}
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) startJob(file);
          }}
        />
      </div>

      {error && <div className="error">{error}</div>}

      {videoUrl && (
        <div className="video-preview">
          {/* Once the mux stage finishes, play the described cut in place of the
              local file — same picture, narration mixed into the soundtrack. The
              key forces a reload when the source swaps. */}
          <video
            key={described ? "described" : "original"}
            src={described && jobId ? describedVideoUrl(jobId) : videoUrl}
            controls
          />
          <p className="video-caption">
            {described
              ? "🎙 With audio description — the original sound ducks under each narration line."
              : "Original audio. The described version replaces this when processing finishes."}
          </p>
        </div>
      )}

      {jobId && (
        <div className="stages">
          {STAGE_ORDER.map((stage) => (
            <span key={stage} className={`stage-pill ${stages[stage] ?? ""}`}>
              {STAGE_LABELS[stage]}
            </span>
          ))}
          <span className="stage-pill">{status ?? ""}</span>
        </div>
      )}

      {allFrames.length > 0 && (
        <p className="frame-count">
          {describedCount} / {allFrames.length} frames described across{" "}
          {orderedSegments.length} shot{orderedSegments.length === 1 ? "" : "s"}
        </p>
      )}

      {orderedSegments.map((seg) => (
        <div className="shot" key={seg.id}>
          <div className="shot-head">
            <span className="shot-title">Shot {seg.id}</span>
            {seg.start != null && seg.end != null ? (
              <span className="shot-range">
                {fmtTime(seg.start)}–{fmtTime(seg.end)}
              </span>
            ) : null}
            <span className="shot-range">
              {seg.frames?.length ?? 0} frame
              {(seg.frames?.length ?? 0) === 1 ? "" : "s"}
            </span>
            {seg.ad_eligible ? (
              <span className="badge ad">
                AD gap {seg.narratable_gap_sec?.toFixed(1)}s
              </span>
            ) : null}
          </div>

          {(seg.frames ?? []).map((frame) => (
            <div className="frame" key={frame.index}>
              <div className="frame-thumb">
                {jobId && frame.path ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={frameUrl(jobId, frame.path)}
                    alt={`shot ${seg.id} frame at ${frame.time}s`}
                  />
                ) : (
                  <div className="thumb-placeholder">no frame</div>
                )}
                <div className="frame-time">{fmtTime(frame.time)}</div>
              </div>

              <div className="frame-analysis">
                {frame.visual ? (
                  <>
                    <p className="frame-desc">{frame.visual.description}</p>
                    {frame.visual.actions.length > 0 && (
                      <div className="chip-row">
                        <span className="chip-label">Actions</span>
                        {frame.visual.actions.map((action, i) => (
                          <span className="chip action" key={i}>
                            {action}
                          </span>
                        ))}
                      </div>
                    )}
                    {frame.visual.entities.length > 0 && (
                      <div className="chip-row">
                        <span className="chip-label">Entities</span>
                        {frame.visual.entities.map((entity, i) => (
                          <span className="chip" key={i}>
                            {entity}
                          </span>
                        ))}
                      </div>
                    )}
                    <div className="frame-meta">
                      <span>Setting: {frame.visual.setting}</span>
                      {frame.visual.on_screen_text ? (
                        <span>
                          On-screen text: “{frame.visual.on_screen_text}”
                        </span>
                      ) : null}
                    </div>
                  </>
                ) : (
                  <p className="frame-pending">analyzing…</p>
                )}
              </div>
            </div>
          ))}

          {seg.audio?.transcript ? (
            <p className="seg-dialogue">“{seg.audio.transcript}”</p>
          ) : null}
          {seg.ad_narration ? (
            <div className="narration">🎙 {seg.ad_narration}</div>
          ) : null}
          {seg.ad_narration_audio && jobId ? (
            <div className="narration-audio">
              <audio
                controls
                src={narrationUrl(jobId, seg.ad_narration_audio)}
              />
              {seg.ad_narration_overflow ? (
                <span
                  className="badge overflow"
                  title="Clip runs longer than the gap"
                >
                  overflow
                </span>
              ) : null}
            </div>
          ) : null}
        </div>
      ))}

      {jobId && (
        <div className="qa">
          <h2>Ask about the video</h2>
          <div className="qa-row">
            <input
              type="text"
              placeholder={
                done
                  ? "e.g. What color is the woman's eyes?"
                  : "Available once processing finishes…"
              }
              value={question}
              disabled={!done || asking}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && onAsk()}
            />
            <button
              onClick={onAsk}
              disabled={!done || asking || !question.trim()}
            >
              {asking ? "Thinking…" : "Ask"}
            </button>
          </div>
          {answer && (
            <>
              <div className="answer">
                {answer.answer ??
                  `The agents could not settle on an answer (status: ${answer.status}).`}
              </div>
              <details className="trace">
                <summary>
                  Reasoning trace — {answer.cycles}{" "}
                  {answer.cycles === 1 ? "cycle" : "cycles"} ({answer.status})
                </summary>
                {answer.history.map((entry, i) => (
                  <TraceCard key={i} entry={entry} />
                ))}
              </details>
            </>
          )}
        </div>
      )}
    </main>
  );
}

/** One agent-trajectory step: the action/agent, why it ran, and its output. */
function TraceCard({ entry }: { entry: TraceEntry }) {
  const detail = entry.result ?? entry.answer ?? entry.comment;
  return (
    <div className="trace-entry">
      <div className="trace-action">{entry.action}</div>
      {entry.reason && <div className="trace-reason">{entry.reason}</div>}
      {entry.instruct && (
        <div className="trace-instruct">instruct: {entry.instruct}</div>
      )}
      {detail && (
        <details>
          <summary>output</summary>
          <pre>{detail}</pre>
        </details>
      )}
    </div>
  );
}
