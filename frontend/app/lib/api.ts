import { accessToken } from "./supabase";

// Base URL of the FastAPI backend. Inlined at build time, so set it per
// deployment environment (Vercel env vars), not at runtime.
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export function wsBase(): string {
  return API_BASE.replace(/^http/, "ws");
}

async function authHeaders(): Promise<Record<string, string>> {
  const token = await accessToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { ...(init.headers ?? {}), ...(await authHeaders()) },
  });
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}));
    throw new Error(detail.detail ?? `request failed (${resp.status})`);
  }
  return (await resp.json()) as T;
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
  /** Job-relative blob key, e.g. "frames/shot_0000_00.jpg". */
  key: string;
  /** Presigned URL for the jpg. Present on timelines fetched from the API. */
  url?: string | null;
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
  ad_narration_key: string | null;
  ad_narration_url?: string | null;
  ad_narration_duration_sec: number | null;
  ad_narration_overflow: boolean | null;
}

export interface Timeline {
  job_id: string;
  duration_sec: number;
  segments: Segment[];
  ad_track_key: string | null;
  ad_track_url?: string | null;
  ad_track_duration_sec: number | null;
  described_key: string | null;
  /** Presigned URL of the video with narration mixed in — what the player plays. */
  described_url?: string | null;
}

export type JobStatus =
  | "created"
  | "queued"
  | "processing"
  | "done"
  | "error"
  | "interrupted";

export interface JobDetail {
  id: string;
  status: JobStatus;
  stage: string | null;
  error: string | null;
  timeline: Timeline | null;
}

export type PipelineEvent = (
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
      key: string;
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
  | { type: "described_video"; video: string }
  | { type: "qa_status"; run_id: string; status: string }
  | {
      type: "qa_result";
      run_id: string;
      status: string;
      answer: string | null;
      cycles: number;
    }
  | { type: "ping" }
  | { type: "status"; status: JobStatus; error: string | null }
) & {
  /** Per-job monotonic sequence; track it so a reconnect can resume. */
  seq?: number;
};

// --------------------------------------------------------------------------
// upload
// --------------------------------------------------------------------------

interface CreateJobResponse {
  job_id: string;
  upload_url: string;
  key: string;
}

/**
 * Reserve a job, PUT the video straight to object storage, then start it.
 *
 * The bytes never pass through the API: it hands back a presigned URL and the
 * browser uploads to R2 directly, so a large file doesn't occupy a request or
 * land on the backend's disk. `onProgress` reports upload percent.
 */
export async function uploadVideo(
  file: File,
  onProgress?: (fraction: number) => void,
): Promise<string> {
  const job = await request<CreateJobResponse>("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename: file.name, content_type: file.type }),
  });

  await putWithProgress(job.upload_url, file, onProgress);
  await request(`/api/jobs/${job.job_id}/start`, { method: "POST" });
  return job.job_id;
}

/**
 * Queue a job from a YouTube URL. A worker downloads the video itself.
 *
 * There is no `/start` call to follow: that step exists to confirm an upload
 * landed in the bucket, and there is no upload here — the job is queued the
 * moment it is created.
 */
export async function submitVideoUrl(url: string): Promise<string> {
  const job = await request<{ job_id: string }>("/api/jobs/from-url", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  return job.job_id;
}

/** XHR rather than fetch, because fetch cannot report upload progress. */
function putWithProgress(
  url: string,
  file: File,
  onProgress?: (fraction: number) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", url);
    if (file.type) xhr.setRequestHeader("Content-Type", file.type);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        onProgress(event.loaded / event.total);
      }
    };
    xhr.onload = () =>
      xhr.status >= 200 && xhr.status < 300
        ? resolve()
        : reject(new Error(`upload failed (${xhr.status})`));
    xhr.onerror = () => reject(new Error("upload failed"));
    xhr.send(file);
  });
}

// --------------------------------------------------------------------------
// jobs
// --------------------------------------------------------------------------

export function getJob(jobId: string): Promise<JobDetail> {
  return request<JobDetail>(`/api/jobs/${jobId}`);
}

/**
 * A websocket URL for a job's event stream, resuming after `sinceSeq`.
 *
 * The ticket is a short-lived single-use credential: a browser can't set an
 * Authorization header on a WebSocket, and putting the session JWT in the URL
 * would leak it into access logs and browser history.
 */
export async function eventsUrl(
  jobId: string,
  sinceSeq: number,
): Promise<string> {
  const { ticket } = await request<{ ticket: string }>(
    `/api/jobs/${jobId}/ws-ticket`,
    { method: "POST" },
  );
  return `${wsBase()}/api/jobs/${jobId}/events?ticket=${encodeURIComponent(
    ticket,
  )}&since=${sinceSeq}`;
}

// --------------------------------------------------------------------------
// Q&A
// --------------------------------------------------------------------------

/** One entry of the Q&A agent trajectory: which agent ran (or a reflection /
 * finish marker), why, and what it returned. Mirrors qa.QAResult history. */
export interface TraceEntry {
  action: string;
  reason?: string;
  instruct?: string;
  result?: string;
  answer?: string;
  assessment?: string;
  comment?: string;
  [k: string]: unknown;
}

export interface QaRun {
  id: string;
  job_id: string;
  question: string;
  status: "queued" | "running" | "completed" | "failed";
  answer: string | null;
  reason: string | null;
  cycles: number | null;
  history: TraceEntry[] | null;
  error: string | null;
}

/** Queue a question. A run takes minutes, so it is not answered inline. */
export async function askQuestion(
  jobId: string,
  question: string,
): Promise<string> {
  const { run_id } = await request<{ run_id: string }>(
    `/api/jobs/${jobId}/ask`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    },
  );
  return run_id;
}

export function getQaRun(runId: string): Promise<QaRun> {
  return request<QaRun>(`/api/qa/${runId}`);
}

/** Presign a batch of this job's blob keys, for <img>/<audio>/<video> sources. */
export async function mediaUrls(
  jobId: string,
  keys: string[],
): Promise<Record<string, string>> {
  const { urls } = await request<{ urls: Record<string, string> }>(
    `/api/jobs/${jobId}/media-urls`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ keys }),
    },
  );
  return urls;
}
