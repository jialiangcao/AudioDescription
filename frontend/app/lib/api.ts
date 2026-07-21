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

export interface VisualAnalysis {
  description: string;
  entities: string[];
  setting: string;
  on_screen_text: string | null;
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
  keyframe: string;
  visual: VisualAnalysis;
  audio: AudioAnalysis | null;
  ad_eligible: boolean | null;
  narratable_gap_sec: number | null;
  ad_narration: string | null;
}

export interface Timeline {
  video_id: string;
  duration_sec: number;
  segments: Segment[];
}

export type JobStatus =
  | "queued"
  | "processing"
  | "done"
  | "error"
  | "interrupted";

export type PipelineEvent =
  | { type: "stage"; stage: string; status: string; [k: string]: unknown }
  | { type: "shot"; shot: Partial<Segment> & { id: number } }
  | { type: "timeline"; timeline: Timeline }
  | { type: "narration"; segment_id: number; text: string }
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
