"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  askQuestion,
  frameUrl,
  narrationUrl,
  uploadVideo,
  wsBase,
  type JobStatus,
  type PipelineEvent,
  type Segment,
} from "./lib/api";

const STAGE_ORDER = [
  "segmentation",
  "vision",
  "audio",
  "timeline",
  "narration",
  "tts",
];
const STAGE_LABELS: Record<string, string> = {
  segmentation: "Shots",
  vision: "Vision",
  audio: "Audio",
  timeline: "Timeline",
  narration: "Narration",
  tts: "Voice",
};

export default function Home() {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [stages, setStages] = useState<Record<string, "active" | "done">>({});
  const [segments, setSegments] = useState<Record<number, Partial<Segment>>>(
    {},
  );
  const [adTrack, setAdTrack] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<string | null>(null);
  const [asking, setAsking] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const startJob = useCallback(async (file: File) => {
    setError(null);
    setAnswer(null);
    setSegments({});
    setStages({});
    setAdTrack(null);
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
        case "shot":
          setSegments((prev) => ({
            ...prev,
            [event.shot.id]: { ...prev[event.shot.id], ...event.shot },
          }));
          break;
        case "timeline":
          setSegments((prev) => {
            const next = { ...prev };
            for (const seg of event.timeline.segments) {
              next[seg.id] = { ...next[seg.id], ...seg };
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
        case "ad_track":
          setAdTrack(event.audio);
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
          <video src={videoUrl} controls />
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

      {orderedSegments.map((seg) => (
        <div className="segment" key={seg.id}>
          {seg.keyframe ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={frameUrl(jobId!, seg.keyframe)}
              alt={`shot ${seg.id}`}
            />
          ) : (
            <div className="thumb-placeholder">analyzing…</div>
          )}
          <div>
            <div className="seg-time">
              Shot {seg.id}
              {seg.start != null && seg.end != null
                ? ` · ${seg.start.toFixed(1)}s–${seg.end.toFixed(1)}s`
                : ""}
              {seg.ad_eligible ? (
                <span className="badge ad" style={{ marginLeft: 8 }}>
                  AD gap {seg.narratable_gap_sec?.toFixed(1)}s
                </span>
              ) : null}
            </div>
            <p className="seg-desc">
              {seg.visual?.description ?? "…"}
            </p>
            {seg.audio?.transcript ? (
              <p className="seg-dialogue">“{seg.audio.transcript}”</p>
            ) : null}
            {seg.ad_narration ? (
              <div className="narration">🎙 {seg.ad_narration}</div>
            ) : null}
            {seg.ad_narration_audio ? (
              <div className="narration-audio">
                <audio
                  controls
                  src={narrationUrl(jobId!, seg.ad_narration_audio)}
                />
                {seg.ad_narration_overflow ? (
                  <span className="badge overflow" title="Clip runs longer than the gap">
                    overflow
                  </span>
                ) : null}
              </div>
            ) : null}
          </div>
        </div>
      ))}

      {adTrack && (
        <div className="ad-track">
          <h2>Full audio-description track</h2>
          <p className="subtitle">
            Every narration line stitched together and spaced to play in sync
            with the video.
          </p>
          <audio controls src={narrationUrl(jobId!, adTrack)} />
        </div>
      )}

      {jobId && (
        <div className="qa">
          <h2>Ask about the video</h2>
          <div className="qa-row">
            <input
              type="text"
              placeholder={
                done ? "e.g. What color is the woman's eyes?" : "Available once processing finishes…"
              }
              value={question}
              disabled={!done || asking}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && onAsk()}
            />
            <button onClick={onAsk} disabled={!done || asking || !question.trim()}>
              {asking ? "Thinking…" : "Ask"}
            </button>
          </div>
          {answer && <div className="answer">{answer}</div>}
        </div>
      )}
    </main>
  );
}
