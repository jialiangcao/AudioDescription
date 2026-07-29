// Base URL of the FastAPI backend. Override in .env.local via
// NEXT_PUBLIC_API_BASE (e.g. when the backend runs on another host).
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export function wsBase(): string {
  return API_BASE.replace(/^http/, "ws");
}

/** Filename portion of a server-side keyframe path, for the frames endpoint. */
export function basename(path: string): string {
  return path.split(/[\\/]/).pop() ?? path;
}

export function frameUrl(jobId: string, keyframePath: string): string {
  return `${API_BASE}/api/jobs/${jobId}/frames/${basename(keyframePath)}`;
}

export function narrationUrl(jobId: string, audioPath: string): string {
  return `${API_BASE}/api/jobs/${jobId}/narration/${basename(audioPath)}`;
}

/** The vision model's analysis of one sampled frame, on its own. */
export interface FrameAnalysis {
  description: string;
  entities: string[];
  actions: string[];
  setting: string;
  on_screen_text: string | null;
}

export interface Frame {
  /** Position of the frame within its shot. */
  index: number;
  /** Absolute timestamp in the video, in seconds. */
  time: number;
  /** Server-side path (or bare filename); pass through frameUrl(). */
  path: string;
  /** null until this frame's vision call lands. */
  visual: FrameAnalysis | null;
}

export interface AudioAnalysis {
  has_speech: boolean;
  transcript: string | null;
  silence_ratio: number;
}

export interface Segment {
  id: number;
  start: number;
  end: number;
  /** Every frame sampled within the shot, in temporal order. */
  frames: Frame[];
  audio: AudioAnalysis | null;
  ad_eligible: boolean | null;
  narratable_gap_sec: number | null;
  narration_start_sec: number | null;
  ad_narration: string | null;
  // Server-side path to the synthesized narration WAV; pass through narrationUrl().
  ad_narration_audio: string | null;
  ad_narration_duration_sec: number | null;
  ad_narration_overflow: boolean | null;
}

export interface Timeline {
  video_id: string;
  duration_sec: number;
  segments: Segment[];
  // Combined AD-only track: every narration clip placed at its play time.
  ad_track_audio: string | null;
  ad_track_duration_sec: number | null;
}

export type JobStatus =
  | "queued"
  | "processing"
  | "done"
  | "error"
  | "interrupted";

export type PipelineEvent =
  | { type: "stage"; stage: string; status: string; [k: string]: unknown }
  // The shot/frame skeleton, sent as soon as frames are extracted — before any
  // frame has been described — so timestamps and images can render immediately.
  | {
      type: "shots";
      shots: {
        id: number;
        start: number;
        end: number;
        frames: Omit<Frame, "visual">[];
      }[];
    }
  // One frame's analysis. Arrives out of order; key off shot_id + index.
  | {
      type: "frame";
      shot_id: number;
      index: number;
      time: number;
      path: string;
      visual: FrameAnalysis | null;
    }
  | { type: "timeline"; timeline: Timeline }
  | { type: "narration"; segment_id: number; text: string }
  | {
      type: "narration_audio";
      segment_id: number;
      audio: string;
      duration_sec: number | null;
      overflow: boolean | null;
    }
  | { type: "ad_track"; audio: string; duration_sec: number | null }
  | { type: "status"; status: JobStatus; error: string | null };

export async function uploadVideo(file: File): Promise<string> {
  const form = new FormData();
  form.append("file", file);
  const resp = await fetch(`${API_BASE}/api/jobs`, {
    method: "POST",
    body: form,
  });
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}));
    throw new Error(detail.detail ?? `upload failed (${resp.status})`);
  }
  return (await resp.json()).job_id;
}

export async function askQuestion(
  jobId: string,
  question: string,
): Promise<string> {
  const resp = await fetch(`${API_BASE}/api/jobs/${jobId}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}));
    throw new Error(detail.detail ?? `request failed (${resp.status})`);
  }
  return (await resp.json()).answer;
}
