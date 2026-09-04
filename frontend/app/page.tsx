"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import AccountMenu from "./components/AccountMenu";
import { useSpeechInput } from "./lib/speech";
import {
    askQuestion,
    deleteJob,
    eventsUrl,
    getJob,
    getQaRun,
    listJobs,
    mediaUrls,
    submitVideoUrl,
    uploadVideo,
    type Frame,
    type JobStatus,
    type JobSummary,
    type PipelineEvent,
    type QaRun,
    type Segment,
    type TraceEntry,
} from "./lib/api";

// Reconnect backoff for the event stream: quick first, then backing off so a
// backend restart doesn't turn into a reconnect storm.
const RECONNECT_DELAYS_MS = [500, 1000, 2000, 5000, 10000];

// How often to poll a queued/running Q&A run. The qa_result event usually
// arrives first; this is the fallback if the socket is between reconnects.
const QA_POLL_MS = 2000;

// How often to re-list the history while some job is still in flight. Only the
// job being watched streams events; the rest of the list has to be polled.
const JOBS_POLL_MS = 5000;

/** Statuses that mean a job is still moving, so the history keeps polling. */
const ACTIVE_STATUSES: JobStatus[] = ["created", "queued", "processing"];

const STAGE_ORDER = [
    "segmentation",
    "vision",
    "audio",
    "timeline",
    "narration",
    "tts",
    "mux",
];

/** Pipeline stage names, said the way a viewer would say them. */
const STAGE_LABELS: Record<string, string> = {
    segmentation: "Splitting into scenes",
    vision: "Watching the video",
    audio: "Listening to the sound",
    timeline: "Finding room to speak",
    narration: "Writing the description",
    tts: "Recording the voice",
    mux: "Mixing the audio",
};

/** Short form, for the step strip under the player. */
const STEP_LABELS: Record<string, string> = {
    segmentation: "Scenes",
    vision: "Watching",
    audio: "Sound",
    timeline: "Timing",
    narration: "Writing",
    tts: "Voice",
    mux: "Mixing",
};

const STATUS_LABELS: Record<JobStatus, string> = {
    created: "Waiting",
    queued: "Waiting",
    processing: "Working",
    done: "Ready",
    error: "Didn’t finish",
    interrupted: "Stopped",
};

/** The Q&A agents, named for what they actually did rather than their class. */
const TRACE_LABELS: Record<string, string> = {
    LocalizeAgent: "Found the moment in the video",
    PerceptionAgent: "Looked closely at the picture",
    SubtitleAgent: "Read what was said",
    reflection: "Double-checked the answer",
    finish: "Settled on an answer",
    parse_failure: "Retried a step",
    unknown_agent: "Retried a step",
};

/** What a question is doing right now, one label per step of the search. */
const QA_STEP_LABELS: Record<string, string> = {
    queued: "Getting ready",
    plan: "Working out what to check",
    localize: "Finding the moment in the video",
    perception: "Looking closely at the picture",
    subtitle: "Reading what was said",
    unknown_agent: "Trying another approach",
    finish: "Putting the answer together",
};

/** Starter questions, so the panel shows what it is for before anyone types. */
const SAMPLE_QUESTIONS = [
    "What is the person wearing?",
    "Where does this take place?",
    "What happens at the end?",
];

/**
 * Collapse the per-stage event log into one bar: how far along the job is, and
 * what it is doing right now. The stages are weighted equally — they are not
 * equally long, but nothing upstream reports intra-stage progress, so an even
 * split is the only honest reading. A stage that has started but not finished
 * counts as half, which keeps the bar moving between events.
 */
function stageProgress(
    status: JobStatus | null,
    stages: Record<string, "active" | "done">,
): { label: string; fraction: number; state: "" | "done" | "error" } {
    if (status === "done") {
        return { label: "Your described video is ready", fraction: 1, state: "done" };
    }
    if (status === "error" || status === "interrupted") {
        const failed = STAGE_ORDER.find((stage) => stages[stage] === "active");
        const what = status === "error" ? "Something went wrong" : "Stopped";
        return {
            label: failed
                ? `${what} while ${STAGE_LABELS[failed].toLowerCase()}`
                : what,
            fraction: 1,
            state: "error",
        };
    }

    const completed = STAGE_ORDER.filter(
        (stage) => stages[stage] === "done",
    ).length;
    // Several stages run concurrently (vision fans out alongside audio), so the
    // *last* active one in pipeline order is the most informative label.
    const active = [...STAGE_ORDER]
        .reverse()
        .find((stage) => stages[stage] === "active");
    const fraction = (completed + (active ? 0.5 : 0)) / STAGE_ORDER.length;

    const label = active
        ? STAGE_LABELS[active]
        : status === "queued" || status === "created"
            ? "Waiting for a free machine"
            : completed === 0
                ? "Getting started"
                : "Working";
    return { label, fraction, state: "" };
}

/** Absolute video timestamp as m:ss.s. */
function fmtTime(sec: number): string {
    const m = Math.floor(sec / 60);
    const s = sec - m * 60;
    return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

/** A video's length as m:ss, or a dash while it is still unknown. */
function fmtDuration(sec: number | null): string {
    if (sec == null) return "—";
    const m = Math.floor(sec / 60);
    const s = Math.round(sec - m * 60);
    return `${m}:${String(s).padStart(2, "0")}`;
}

/** When a video was added, in the viewer's own locale and timezone. */
function fmtCreated(iso: string): string {
    const date = new Date(iso);
    return Number.isNaN(date.getTime())
        ? iso
        : date.toLocaleString(undefined, {
            month: "short",
            day: "numeric",
            hour: "numeric",
            minute: "2-digit",
        });
}

/**
 * Merge incoming frames into a scene's existing frame list by `index`.
 *
 * Frame analyses arrive out of order and after the scene skeleton, so neither
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
            key: frame.key ?? "",
            visual: frame.visual ?? null,
        }));
}

type Tab = "studio" | "library";

export default function Home() {
    const [tab, setTab] = useState<Tab>("studio");
    const [jobId, setJobId] = useState<string | null>(null);
    const [jobName, setJobName] = useState<string | null>(null);
    const [status, setStatus] = useState<JobStatus | null>(null);
    const [stages, setStages] = useState<Record<string, "active" | "done">>({});
    const [segments, setSegments] = useState<Record<number, Partial<Segment>>>(
        {},
    );
    const [described, setDescribed] = useState(false);
    // The scene-by-scene breakdown is verbose (one card per sampled frame), so
    // it stays collapsed until asked for.
    const [showFrames, setShowFrames] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [dragging, setDragging] = useState(false);
    const [urlInput, setUrlInput] = useState("");

    const [question, setQuestion] = useState("");
    const [answer, setAnswer] = useState<QaRun | null>(null);
    const [asking, setAsking] = useState(false);
    // What the running question is doing right now, and the steps it has already
    // taken — streamed as the search happens, so the panel is never a blank wait.
    const [qaStep, setQaStep] = useState<string | null>(null);
    const [qaTrail, setQaTrail] = useState<TraceEntry[]>([]);
    // The run those belong to, so a late event from a previous question is
    // ignored rather than shown under the new one.
    const qaRunRef = useRef<string | null>(null);
    // The video queued for deletion, and so the subject of the confirm dialog.
    const [confirming, setConfirming] = useState<JobSummary | null>(null);
    const [uploadPct, setUploadPct] = useState<number | null>(null);
    // Every video this user has ever added, newest first.
    const [jobs, setJobs] = useState<JobSummary[]>([]);
    // Id of the video whose delete is in flight, so its button can't fire twice.
    const [deleting, setDeleting] = useState<string | null>(null);
    const [describedUrl, setDescribedUrl] = useState<string | null>(null);
    // The played video's own width/height ratio, read from its metadata, so the
    // player frames it exactly instead of letterboxing it into a fixed 16:9.
    const [aspect, setAspect] = useState<number | null>(null);
    // Blob key -> presigned URL. Progress events carry keys, and <img>/<audio>
    // cannot send an auth header, so URLs are fetched in batches as keys appear.
    const [urls, setUrls] = useState<Record<string, string>>({});
    const fileInputRef = useRef<HTMLInputElement>(null);
    // Last event sequence applied, so a reconnect resumes instead of replaying.
    const lastSeq = useRef(0);

    // Nothing plays until the described cut exists, so there is no preview to
    // set up here — every entry point clears exactly the same state.
    const resetForNewJob = useCallback((name: string | null) => {
        setError(null);
        setAnswer(null);
        setQaStep(null);
        setQaTrail([]);
        qaRunRef.current = null;
        setQuestion("");
        setSegments({});
        setStages({});
        setDescribed(false);
        setDescribedUrl(null);
        setAspect(null);
        setStatus(null);
        setJobName(name);
        setUrls({});
        setShowFrames(false);
        lastSeq.current = 0;
    }, []);

    const refreshJobs = useCallback(async () => {
        try {
            setJobs(await listJobs());
        } catch {
            /* the library is a convenience; a failed refresh keeps the last list */
        }
    }, []);

    /**
     * Show a video that already exists: hydrate whatever it has finished, and
     * let the event stream fill in the rest.
     *
     * A stored timeline is the whole result — frames, description and the
     * described cut, each already presigned — so a finished video renders
     * without waiting on the socket. One still in progress has no timeline yet,
     * but the websocket effect replays its event log from seq 0, which rebuilds
     * exactly what a viewer who had been watching all along would see.
     */
    const openJob = useCallback(
        async (summary: JobSummary) => {
            setTab("studio");
            if (summary.id === jobId) return;
            resetForNewJob(summary.filename ?? "Video from a link");
            setJobId(summary.id);
            setStatus(summary.status);
            try {
                const job = await getJob(summary.id);
                setStatus(job.status);
                if (job.error) setError(job.error);
                const timeline = job.timeline;
                if (!timeline) return;
                setSegments(
                    Object.fromEntries(timeline.segments.map((seg) => [seg.id, seg])),
                );
                // The timeline's URLs are already presigned, so seeding the map here
                // spares the browser a round trip per batch of frames.
                const seeded: Record<string, string> = {};
                for (const seg of timeline.segments) {
                    for (const frame of seg.frames) {
                        if (frame.url) seeded[frame.key] = frame.url;
                    }
                    if (seg.ad_narration_key && seg.ad_narration_url) {
                        seeded[seg.ad_narration_key] = seg.ad_narration_url;
                    }
                }
                setUrls(seeded);
                if (timeline.described_url) {
                    setDescribed(true);
                    setDescribedUrl(timeline.described_url);
                }
            } catch (e) {
                setError(e instanceof Error ? e.message : String(e));
            }
        },
        [jobId, resetForNewJob],
    );

    /**
     * Delete a video for good — the file, its description and its questions.
     *
     * The dialog does the confirming; by the time this runs the answer is yes.
     * There is nothing to undo it with — the source video goes along with
     * everything made from it — so starting over would mean uploading again. If
     * the video being viewed is the one going away, the studio is cleared rather
     * than left pointing at rows that are gone.
     */
    const removeJob = useCallback(
        async (summary: JobSummary) => {
            setConfirming(null);
            setDeleting(summary.id);
            try {
                await deleteJob(summary.id);
                setJobs((prev) => prev.filter((job) => job.id !== summary.id));
                if (summary.id === jobId) {
                    resetForNewJob(null);
                    setJobId(null);
                }
            } catch (e) {
                setError(e instanceof Error ? e.message : String(e));
                void refreshJobs();
            } finally {
                setDeleting(null);
            }
        },
        [jobId, refreshJobs, resetForNewJob],
    );

    /** Leave the open video and go back to the upload screen. */
    const closeJob = useCallback(() => {
        resetForNewJob(null);
        setJobId(null);
        setUploadPct(null);
    }, [resetForNewJob]);

    const startJob = useCallback(
        async (file: File) => {
            setTab("studio");
            resetForNewJob(file.name);
            setJobId(null);
            try {
                setUploadPct(0);
                const id = await uploadVideo(file, (fraction) =>
                    setUploadPct(Math.round(fraction * 100)),
                );
                setUploadPct(null);
                setJobId(id);
                setStatus("queued");
                void refreshJobs();
            } catch (e) {
                setUploadPct(null);
                setError(e instanceof Error ? e.message : String(e));
            }
        },
        [resetForNewJob, refreshJobs],
    );

    const startJobFromUrl = useCallback(
        async (url: string) => {
            setTab("studio");
            resetForNewJob("Video from a link");
            setJobId(null);
            try {
                const id = await submitVideoUrl(url);
                setUrlInput("");
                setJobId(id);
                setStatus("queued");
                void refreshJobs();
            } catch (e) {
                setError(e instanceof Error ? e.message : String(e));
            }
        },
        [resetForNewJob, refreshJobs],
    );

    // Stream live pipeline events while a job is active, reconnecting on drop and
    // resuming from the last sequence seen rather than replaying the whole log.
    useEffect(() => {
        if (!jobId) return;
        let socket: WebSocket | null = null;
        let retry = 0;
        let reconnectTimer: ReturnType<typeof setTimeout> | undefined;
        let closed = false;
        let finished = false;

        const handle = (event: PipelineEvent) => {
            if (typeof event.seq === "number") lastSeq.current = event.seq;
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
                                    key: event.key,
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
                            ad_narration_key: event.audio,
                            ad_narration_duration_sec: event.duration_sec,
                            ad_narration_overflow: event.overflow,
                        },
                    }));
                    break;
                case "described_video":
                    setDescribed(true);
                    break;
                case "qa_status":
                    if (qaRunRef.current && qaRunRef.current !== event.run_id) break;
                    if (event.status === "running") setQaStep("plan");
                    break;
                case "qa_trace":
                    // The node that just finished is the best available read on where
                    // the search has got to.
                    if (qaRunRef.current && qaRunRef.current !== event.run_id) break;
                    setQaStep(event.node);
                    setQaTrail((prev) => [...prev, ...event.records]);
                    break;
                case "qa_result":
                    // The trace panel refreshes from the run itself; this just wakes it.
                    setAnswer((prev) =>
                        prev && prev.id === event.run_id
                            ? { ...prev, status: event.status as QaRun["status"] }
                            : prev,
                    );
                    break;
                case "status":
                    setStatus(event.status);
                    // The server closes the socket after a terminal status, so stop
                    // reconnecting or the close handler would loop forever.
                    if (["done", "error", "interrupted"].includes(event.status)) {
                        finished = true;
                    }
                    if (event.status === "error") {
                        setError(event.error ?? "We couldn’t finish describing this video.");
                    }
                    break;
            }
        };

        const connect = async () => {
            if (closed) return;
            try {
                socket = new WebSocket(await eventsUrl(jobId, lastSeq.current));
            } catch {
                scheduleReconnect();
                return;
            }
            socket.onopen = () => {
                retry = 0;
            };
            socket.onmessage = (msg) => handle(JSON.parse(msg.data));
            // A terminal status closes the socket server-side, so only reconnect
            // while the job is still in flight.
            socket.onclose = () => {
                if (!closed && !finished) scheduleReconnect();
            };
        };

        const scheduleReconnect = () => {
            const delay =
                RECONNECT_DELAYS_MS[Math.min(retry, RECONNECT_DELAYS_MS.length - 1)];
            retry += 1;
            reconnectTimer = setTimeout(connect, delay);
        };

        void connect();
        return () => {
            closed = true;
            clearTimeout(reconnectTimer);
            socket?.close();
        };
    }, [jobId]);

    // Presign any blob keys we've learned about but don't yet have a URL for.
    // Batched, because a long video streams in hundreds of frame keys.
    const orderedSegmentsForUrls = Object.values(segments);
    useEffect(() => {
        if (!jobId) return;
        const wanted = new Set<string>();
        for (const seg of orderedSegmentsForUrls) {
            for (const frame of seg.frames ?? []) {
                if (frame.key) wanted.add(frame.key);
            }
            if (seg.ad_narration_key) wanted.add(seg.ad_narration_key);
        }
        const missing = [...wanted].filter((key) => !urls[key]);
        if (missing.length === 0) return;

        let cancelled = false;
        mediaUrls(jobId, missing)
            .then((fresh) => {
                if (!cancelled) setUrls((prev) => ({ ...prev, ...fresh }));
            })
            .catch(() => {
                /* a missing thumbnail is not worth surfacing as a failure */
            });
        return () => {
            cancelled = true;
        };
    }, [jobId, orderedSegmentsForUrls, urls]);

    // Once the video is finished, fetch the timeline for the described cut's URL.
    useEffect(() => {
        if (!jobId || !described) return;
        let cancelled = false;
        getJob(jobId)
            .then((job) => {
                if (!cancelled) setDescribedUrl(job.timeline?.described_url ?? null);
            })
            .catch(() => {
                if (!cancelled) setError("We couldn’t load the described video.");
            });
        return () => {
            cancelled = true;
        };
    }, [jobId, described]);

    // A question is answered by a worker, so poll it until it settles. The
    // qa_result event usually beats this; the poll covers a dropped socket.
    useEffect(() => {
        const runId = answer?.id;
        if (!runId) return;
        if (answer.status === "completed" || answer.status === "failed") return;
        const timer = setInterval(async () => {
            try {
                setAnswer(await getQaRun(runId));
            } catch {
                /* keep polling; a transient failure is not fatal */
            }
        }, QA_POLL_MS);
        return () => clearInterval(timer);
    }, [answer?.id, answer?.status]);

    // The library: loaded once, then re-listed whenever the open video changes
    // status (its own row is now stale) and on a timer while anything is running.
    useEffect(() => {
        void refreshJobs();
    }, [refreshJobs, status]);

    const anyActive = jobs.some((job) => ACTIVE_STATUSES.includes(job.status));
    useEffect(() => {
        if (!anyActive) return;
        const timer = setInterval(() => void refreshJobs(), JOBS_POLL_MS);
        return () => clearInterval(timer);
    }, [anyActive, refreshJobs]);

    const onDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setDragging(false);
        const file = e.dataTransfer.files?.[0];
        if (file) startJob(file);
    };

    const onAsk = async (text?: string) => {
        const asked = (text ?? question).trim();
        if (!jobId || !asked) return;
        setQuestion(asked);
        setAsking(true);
        setAnswer(null);
        setError(null);
        setQaStep("queued");
        setQaTrail([]);
        qaRunRef.current = null;
        try {
            const runId = await askQuestion(jobId, asked);
            qaRunRef.current = runId;
            setAnswer(await getQaRun(runId));
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setAsking(false);
        }
    };

    const done = status === "done";

    // Speaking a question takes the same path as typing one: the transcript
    // lands in the box and is sent as soon as the sentence settles, so a viewer
    // who is using this because they can't see the screen never has to find the
    // Ask button. Firefox has no Web Speech API, and there `supported` is false
    // and the mic is not rendered at all.
    const speech = useSpeechInput({
        onFinal: (transcript) => {
            setQuestion(transcript);
            if (done && !asking) void onAsk(transcript);
        },
    });

    const micLabel = !speech.supported
        ? "Voice input needs Chrome, Edge or Safari"
        : speech.listening
            ? "Stop listening"
            : "Ask by voice";
    // The unsupported case is stated once the box is usable, so a viewer who
    // can't see a greyed-out button still learns why speaking isn't an option.
    const micNote = speech.error
        ? speech.error
        : speech.listening
            ? "Listening… stop speaking and your question is sent."
            : done && !speech.supported
                ? "This browser can’t listen — Chrome, Edge and Safari can."
                : null;

    const orderedSegments = Object.values(segments).sort(
        (a, b) => (a.id ?? 0) - (b.id ?? 0),
    );
    const allFrames = orderedSegments.flatMap((seg) => seg.frames ?? []);
    const describedCount = allFrames.filter((f) => f.visual).length;
    const failed = status === "error" || status === "interrupted";
    const progress = stageProgress(status, stages);
    // Nothing is playable until the described cut has been mixed and presigned.
    const ready = done && described && describedUrl !== null;
    const busy = uploadPct !== null || (jobId !== null && !done && !failed);
    // A question is in flight from the moment it is sent until the run settles.
    const qaWorking =
        asking || answer?.status === "queued" || answer?.status === "running";

    return (
        <div className="app">
            <header className="topbar">
                <div className="topbar-inner">
                    <span className="wordmark">BuddyWatch</span>
                    <nav className="tabs" aria-label="Sections">
                        <button
                            type="button"
                            className={`tab${tab === "studio" ? " on" : ""}`}
                            aria-current={tab === "studio" ? "page" : undefined}
                            onClick={() => setTab("studio")}
                        >
                            Studio
                        </button>
                        <button
                            type="button"
                            className={`tab${tab === "library" ? " on" : ""}`}
                            aria-current={tab === "library" ? "page" : undefined}
                            onClick={() => setTab("library")}
                        >
                            My videos
                            {jobs.length > 0 && <span className="tab-count">{jobs.length}</span>}
                        </button>
                    </nav>
                    <AccountMenu />
                </div>
            </header>

            {tab === "studio" ? (
                <main className="page">
                    {error && (
                        <p className="notice notice-error" role="alert">
                            {error}
                        </p>
                    )}

                    {jobId || busy ? (
                        <>
                            <button type="button" className="backlink" onClick={closeJob}>
                                <BackIcon />
                                Add another video
                            </button>

                            <div className="workspace">
                                <section className="stage-col" aria-label="Your video">
                                    <div className="stage-head">
                                        <h1 className="stage-title">
                                            {jobName ?? "Your video"}
                                        </h1>
                                        {status && (
                                            <span className={`pill ${status}`}>
                                                {STATUS_LABELS[status]}
                                            </span>
                                        )}
                                    </div>

                                    <div
                                        className={`screen${ready ? " live" : ""}`}
                                        style={
                                            aspect
                                                ? {
                                                    aspectRatio: aspect,
                                                    maxWidth: `calc(72vh * ${aspect})`,
                                                }
                                                : undefined
                                        }
                                    >
                                        {ready ? (
                                            <video
                                                src={describedUrl}
                                                controls
                                                playsInline
                                                onLoadedMetadata={(event) => {
                                                    const el = event.currentTarget;
                                                    if (!el.videoWidth || !el.videoHeight) return;
                                                    // Clamp: a very tall or very wide source still has to
                                                    // sit in the column without pushing the page around.
                                                    setAspect(
                                                        Math.min(
                                                            Math.max(el.videoWidth / el.videoHeight, 0.5),
                                                            3,
                                                        ),
                                                    );
                                                }}
                                            />
                                        ) : (
                                            <div className="screen-wait">
                                                {failed ? (
                                                    <>
                                                        <p className="screen-headline">
                                                            We couldn’t finish this one
                                                        </p>
                                                        <p className="screen-sub">
                                                            Nothing was lost — add the video again to retry.
                                                        </p>
                                                    </>
                                                ) : done ? (
                                                    <>
                                                        <p className="screen-headline">
                                                            Nothing to describe here
                                                        </p>
                                                        <p className="screen-sub">
                                                            There were no quiet gaps long enough to speak in,
                                                            so we left the soundtrack alone.
                                                        </p>
                                                    </>
                                                ) : (
                                                    <>
                                                        <SoundMark />
                                                        <p className="screen-headline">
                                                            {uploadPct !== null
                                                                ? "Uploading your video…"
                                                                : "Writing your description"}
                                                        </p>
                                                        <p className="screen-sub">
                                                            Your video will play here, with the description
                                                            spoken over it, as soon as it’s ready.
                                                        </p>
                                                    </>
                                                )}
                                            </div>
                                        )}
                                    </div>

                                    {uploadPct !== null ? (
                                        <div className="progress">
                                            <div className="progress-row">
                                                <span className="progress-stage">Uploading</span>
                                                <span className="progress-pct">{uploadPct}%</span>
                                            </div>
                                            <div
                                                className="progress-track"
                                                role="progressbar"
                                                aria-valuemin={0}
                                                aria-valuemax={100}
                                                aria-valuenow={uploadPct}
                                            >
                                                <div
                                                    className="progress-fill"
                                                    style={{ width: `${uploadPct}%` }}
                                                />
                                            </div>
                                        </div>
                                    ) : jobId ? (
                                        <>
                                            <div className="progress">
                                                <div className="progress-row">
                                                    <span className="progress-stage">
                                                        {progress.label}
                                                    </span>
                                                    <span className="progress-pct">
                                                        {Math.round(progress.fraction * 100)}%
                                                    </span>
                                                </div>
                                                <div
                                                    className="progress-track"
                                                    role="progressbar"
                                                    aria-valuemin={0}
                                                    aria-valuemax={100}
                                                    aria-valuenow={Math.round(progress.fraction * 100)}
                                                    aria-valuetext={progress.label}
                                                >
                                                    <div
                                                        className={`progress-fill ${progress.state}`}
                                                        style={{ width: `${progress.fraction * 100}%` }}
                                                    />
                                                </div>
                                            </div>

                                            <ol className="steps">
                                                {STAGE_ORDER.map((stage) => (
                                                    <li
                                                        key={stage}
                                                        className={`step ${done ? "done" : (stages[stage] ?? "")
                                                            }`}
                                                    >
                                                        {STEP_LABELS[stage]}
                                                    </li>
                                                ))}
                                            </ol>
                                        </>
                                    ) : null}
                                </section>

                                <section className="rail" aria-label="Ask about this video">
                                    <h2 className="rail-title">Ask about this video</h2>

                                    <div className="ask">
                                        <input
                                            type="text"
                                            placeholder={
                                                done
                                                    ? speech.supported
                                                        ? "Type or speak your question"
                                                        : "What colour is her jacket?"
                                                    : "Ready once your video is described"
                                            }
                                            aria-label="Your question"
                                            // While the mic is open the box shows what is being
                                            // heard; the words only become the question once the
                                            // sentence settles.
                                            value={
                                                speech.listening && speech.interim
                                                    ? speech.interim
                                                    : question
                                            }
                                            disabled={!done || asking || speech.listening}
                                            onChange={(e) => setQuestion(e.target.value)}
                                            onKeyDown={(e) => e.key === "Enter" && void onAsk()}
                                        />
                                        {/* Always rendered, even where the browser has no
                                            speech API: a mic that silently isn't there is
                                            indistinguishable from one that is broken, so it
                                            shows disabled and says why instead. */}
                                        <button
                                            type="button"
                                            className={`btn-mic${speech.listening ? " on" : ""}`}
                                            onClick={speech.toggle}
                                            disabled={!done || asking || !speech.supported}
                                            aria-pressed={speech.listening}
                                            aria-label={micLabel}
                                            title={micLabel}
                                        >
                                            <MicIcon />
                                        </button>
                                        <button
                                            type="button"
                                            className="btn-primary"
                                            onClick={() => void onAsk()}
                                            disabled={!done || asking || !question.trim()}
                                        >
                                            {asking ? "Looking…" : "Ask"}
                                        </button>
                                    </div>

                                    {micNote && (
                                        <p
                                            className={`mic-note${speech.error ? " bad" : ""}`}
                                            role="status"
                                            aria-live="polite"
                                        >
                                            {micNote}
                                        </p>
                                    )}

                                    {done && !answer && !qaWorking && (
                                        <div className="samples">
                                            <span className="samples-label">Try</span>
                                            {SAMPLE_QUESTIONS.map((sample) => (
                                                <button
                                                    key={sample}
                                                    type="button"
                                                    className="sample"
                                                    onClick={() => void onAsk(sample)}
                                                >
                                                    {sample}
                                                </button>
                                            ))}
                                        </div>
                                    )}

                                    {qaWorking && (
                                        <div className="thinking" role="status" aria-live="polite">
                                            <div className="thinking-head">
                                                <ThinkingDots />
                                                <span className="thinking-step">
                                                    {QA_STEP_LABELS[qaStep ?? "queued"] ??
                                                        "Searching the video"}
                                                </span>
                                            </div>
                                            <p className="thinking-sub">
                                                {qaTrail.length > 0
                                                    ? `${qaTrail.length} ${qaTrail.length === 1 ? "step" : "steps"
                                                    } so far — this usually takes a minute or two.`
                                                    : "This usually takes a minute or two."}
                                            </p>
                                            {qaTrail.length > 0 && (
                                                <ul className="thinking-trail">
                                                    {qaTrail.slice(-4).map((entry, i) => (
                                                        <li key={qaTrail.length - 4 + i}>
                                                            {TRACE_LABELS[entry.action] ?? entry.action}
                                                        </li>
                                                    ))}
                                                </ul>
                                            )}
                                        </div>
                                    )}

                                    {answer && !qaWorking && (
                                        <>
                                            <div className="answer">
                                                {answer.answer ??
                                                    answer.error ??
                                                    "We couldn’t find an answer in this video."}
                                            </div>
                                            {answer.history && answer.history.length > 0 && (
                                                <details className="trace">
                                                    <summary>
                                                        How we worked it out — {answer.cycles ?? 0}{" "}
                                                        {answer.cycles === 1 ? "step" : "steps"}
                                                    </summary>
                                                    {answer.history.map((entry, i) => (
                                                        <TraceCard key={i} entry={entry} />
                                                    ))}
                                                </details>
                                            )}
                                        </>
                                    )}
                                </section>
                            </div>

                            {allFrames.length > 0 && (
                                <section className="breakdown" aria-label="What we saw">
                                    <div className="breakdown-head">
                                        <div>
                                            <h2 className="section-title">What we saw</h2>
                                            <p className="section-sub">
                                                {describedCount} of {allFrames.length} moments looked at,
                                                across {orderedSegments.length}{" "}
                                                {orderedSegments.length === 1 ? "scene" : "scenes"}
                                            </p>
                                        </div>
                                        <button
                                            type="button"
                                            className="btn-toggle"
                                            aria-expanded={showFrames}
                                            onClick={() => setShowFrames((prev) => !prev)}
                                        >
                                            <span aria-hidden="true" className="btn-toggle-caret">
                                                ▸
                                            </span>
                                            {showFrames
                                                ? "Hide frame-by-frame"
                                                : "Show frame-by-frame"}
                                        </button>
                                    </div>

                                    {showFrames &&
                                        orderedSegments.map((seg) => (
                                            <article className="scene" key={seg.id}>
                                                <header className="scene-head">
                                                    <span className="scene-title">
                                                        Scene {(seg.id ?? 0) + 1}
                                                    </span>
                                                    {seg.start != null && seg.end != null ? (
                                                        <span className="scene-range">
                                                            {fmtTime(seg.start)}–{fmtTime(seg.end)}
                                                        </span>
                                                    ) : null}
                                                    <span className="scene-range">
                                                        {seg.frames?.length ?? 0}{" "}
                                                        {(seg.frames?.length ?? 0) === 1
                                                            ? "moment"
                                                            : "moments"}
                                                    </span>
                                                    {seg.ad_eligible ? (
                                                        <span className="tag">
                                                            {seg.narratable_gap_sec?.toFixed(1)}s of quiet to
                                                            speak in
                                                        </span>
                                                    ) : null}
                                                </header>

                                                {(seg.frames ?? []).map((frame) => (
                                                    <div className="moment" key={frame.index}>
                                                        <div className="moment-thumb">
                                                            {urls[frame.key] ? (
                                                                // eslint-disable-next-line @next/next/no-img-element
                                                                <img
                                                                    src={urls[frame.key]}
                                                                    alt={`Scene ${(seg.id ?? 0) + 1} at ${fmtTime(frame.time)}`}
                                                                />
                                                            ) : (
                                                                <div className="thumb-placeholder" />
                                                            )}
                                                            <div className="moment-time">
                                                                {fmtTime(frame.time)}
                                                            </div>
                                                        </div>

                                                        <div className="moment-body">
                                                            {frame.visual ? (
                                                                <>
                                                                    <p className="moment-desc">
                                                                        {frame.visual.description}
                                                                    </p>
                                                                    {frame.visual.actions.length > 0 && (
                                                                        <div className="chip-row">
                                                                            <span className="chip-label">
                                                                                Happening
                                                                            </span>
                                                                            {frame.visual.actions.map((action, i) => (
                                                                                <span className="chip action" key={i}>
                                                                                    {action}
                                                                                </span>
                                                                            ))}
                                                                        </div>
                                                                    )}
                                                                    {frame.visual.entities.length > 0 && (
                                                                        <div className="chip-row">
                                                                            <span className="chip-label">
                                                                                On screen
                                                                            </span>
                                                                            {frame.visual.entities.map(
                                                                                (entity, i) => (
                                                                                    <span className="chip" key={i}>
                                                                                        {entity}
                                                                                    </span>
                                                                                ),
                                                                            )}
                                                                        </div>
                                                                    )}
                                                                    <div className="moment-meta">
                                                                        <span>Where: {frame.visual.setting}</span>
                                                                        {frame.visual.on_screen_text ? (
                                                                            <span>
                                                                                Text: “{frame.visual.on_screen_text}”
                                                                            </span>
                                                                        ) : null}
                                                                    </div>
                                                                </>
                                                            ) : (
                                                                <p className="moment-pending">Looking…</p>
                                                            )}
                                                        </div>
                                                    </div>
                                                ))}

                                                {seg.audio?.transcript ? (
                                                    <p className="scene-dialogue">
                                                        “{seg.audio.transcript}”
                                                    </p>
                                                ) : null}
                                                {seg.ad_narration ? (
                                                    <div className="scene-narration">
                                                        <span className="scene-narration-label">
                                                            Description
                                                        </span>
                                                        <p>{seg.ad_narration}</p>
                                                    </div>
                                                ) : null}
                                                {seg.ad_narration_key && urls[seg.ad_narration_key] ? (
                                                    <div className="scene-audio">
                                                        <audio controls src={urls[seg.ad_narration_key]} />
                                                        {seg.ad_narration_overflow ? (
                                                            <span
                                                                className="tag warn"
                                                                title="This line runs past the quiet gap"
                                                            >
                                                                Runs long
                                                            </span>
                                                        ) : null}
                                                    </div>
                                                ) : null}
                                            </article>
                                        ))}
                                </section>
                            )}
                        </>
                    ) : (
                        <section className="welcome">
                            <h1 className="welcome-title">
                                Video that describes itself, for blind and low-vision viewers
                            </h1>
                            <Intake
                                dragging={dragging}
                                setDragging={setDragging}
                                onDrop={onDrop}
                                fileInputRef={fileInputRef}
                                onFile={startJob}
                                urlInput={urlInput}
                                setUrlInput={setUrlInput}
                                onUrl={startJobFromUrl}
                            />
                            <ol className="how">
                                <li>
                                    <span className="how-step">Watch</span>
                                    Every scene is looked at moment by moment.
                                </li>
                                <li>
                                    <span className="how-step">Write</span>
                                    A line of description is written to fit each quiet gap.
                                </li>
                                <li>
                                    <span className="how-step">Speak</span>
                                    It&rsquo;s read aloud and mixed under the original sound.
                                </li>
                            </ol>
                            <div className="how-close">
                                <p className="how-claim">Ready in minutes, on demand</p>
                                <p className="how-note">
                                    Description is normally written and voiced by hand, days
                                    per title. Here it runs the moment you upload.
                                </p>
                            </div>
                        </section>
                    )}
                </main>
            ) : (
                <main className="page">
                    <div className="library-head">
                        <div>
                            <h1 className="section-title">My videos</h1>
                            <p className="section-sub">
                                Everything you&rsquo;ve added. Open one to watch it or ask about
                                it.
                            </p>
                        </div>
                        <button
                            type="button"
                            className="btn-primary"
                            onClick={() => setTab("studio")}
                        >
                            Add a video
                        </button>
                    </div>

                    {jobs.length === 0 ? (
                        <div className="empty">
                            <p className="empty-headline">No videos yet</p>
                            <p className="empty-copy">
                                Add your first one and it&rsquo;ll show up here.
                            </p>
                        </div>
                    ) : (
                        <ul className="library">
                            {jobs.map((job) => {
                                const active = ACTIVE_STATUSES.includes(job.status);
                                const stageLabel =
                                    job.stage && STAGE_LABELS[job.stage]
                                        ? STAGE_LABELS[job.stage]
                                        : STATUS_LABELS[job.status];
                                return (
                                    <li
                                        key={job.id}
                                        className={`library-row${job.id === jobId ? " current" : ""}`}
                                    >
                                        {/* The row and its delete control are siblings, not nested
                        buttons: a button inside a button is invalid HTML, and
                        deleting must not also open the video. */}
                                        <button
                                            type="button"
                                            className="library-open"
                                            onClick={() => void openJob(job)}
                                        >
                                            <span className="library-name">
                                                {job.filename ?? "Video from a link"}
                                            </span>
                                            <span className={`pill ${job.status}`}>
                                                {active ? stageLabel : STATUS_LABELS[job.status]}
                                            </span>
                                            <span className="library-meta">
                                                {fmtDuration(job.duration_sec)}
                                            </span>
                                            <span className="library-meta">
                                                {fmtCreated(job.created_at)}
                                            </span>
                                        </button>
                                        <button
                                            type="button"
                                            className="library-delete"
                                            title="Delete this video"
                                            aria-label={`Delete ${job.filename ?? "this video"}`}
                                            disabled={deleting === job.id}
                                            onClick={() => setConfirming(job)}
                                        >
                                            <TrashIcon />
                                        </button>
                                    </li>
                                );
                            })}
                        </ul>
                    )}
                </main>
            )}

            {confirming && (
                <ConfirmDelete
                    job={confirming}
                    onCancel={() => setConfirming(null)}
                    onConfirm={() => void removeJob(confirming)}
                />
            )}
        </div>
    );
}

/** The two ways in: drop a file, or paste a link a worker fetches for you. */
function Intake({
    dragging,
    setDragging,
    onDrop,
    fileInputRef,
    onFile,
    urlInput,
    setUrlInput,
    onUrl,
}: {
    dragging: boolean;
    setDragging: (value: boolean) => void;
    onDrop: (e: React.DragEvent) => void;
    fileInputRef: React.RefObject<HTMLInputElement | null>;
    onFile: (file: File) => void;
    urlInput: string;
    setUrlInput: (value: string) => void;
    onUrl: (url: string) => void;
}) {
    return (
        <div className="intake">
            <div
                className={`dropzone${dragging ? " drag" : ""}`}
                role="button"
                tabIndex={0}
                onDragOver={(e) => {
                    e.preventDefault();
                    setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                onClick={() => fileInputRef.current?.click()}
                onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        fileInputRef.current?.click();
                    }
                }}
            >
                <span className="dropzone-main">Drop a video here</span>
                <span className="dropzone-sub">or click to choose a file</span>
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="video/*"
                    hidden
                    onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) onFile(file);
                    }}
                />
            </div>

            <div className="intake-or" role="separator" aria-orientation="horizontal">
                <span>or</span>
            </div>

            <form
                className="urlbar"
                onSubmit={(e) => {
                    e.preventDefault();
                    const url = urlInput.trim();
                    if (url) onUrl(url);
                }}
            >
                <input
                    type="url"
                    placeholder="Paste a YouTube link"
                    aria-label="YouTube link"
                    value={urlInput}
                    onChange={(e) => setUrlInput(e.target.value)}
                />
                <button type="submit" className="btn-primary" disabled={!urlInput.trim()}>
                    Add
                </button>
            </form>
        </div>
    );
}

/**
 * Confirms a delete in the page rather than in a browser alert.
 *
 * Deleting takes the video and everything made from it, and there is no undo,
 * so the dialog says exactly that and keeps the safe choice under the initial
 * focus. Escape and a click outside both mean "no".
 */
function ConfirmDelete({
    job,
    onCancel,
    onConfirm,
}: {
    job: JobSummary;
    onCancel: () => void;
    onConfirm: () => void;
}) {
    const cancelRef = useRef<HTMLButtonElement>(null);

    useEffect(() => {
        cancelRef.current?.focus();
        const onKey = (e: KeyboardEvent) => {
            if (e.key === "Escape") onCancel();
        };
        document.addEventListener("keydown", onKey);
        return () => document.removeEventListener("keydown", onKey);
    }, [onCancel]);

    const name = job.filename ?? "this video";
    const running = ACTIVE_STATUSES.includes(job.status);

    return (
        <div
            className="modal-backdrop"
            onMouseDown={(e) => {
                if (e.target === e.currentTarget) onCancel();
            }}
        >
            <div
                className="modal"
                role="dialog"
                aria-modal="true"
                aria-labelledby="confirm-delete-title"
            >
                <h2 className="modal-title" id="confirm-delete-title">
                    Delete {name}?
                </h2>
                <p className="modal-copy">
                    {running
                        ? "It’s still being described. Deleting stops that and removes the video, its description and everything you’ve asked about it. This can’t be undone."
                        : "This removes the video, its description and everything you’ve asked about it. This can’t be undone."}
                </p>
                <div className="modal-actions">
                    <button
                        type="button"
                        className="btn-quiet"
                        ref={cancelRef}
                        onClick={onCancel}
                    >
                        Keep it
                    </button>
                    <button type="button" className="btn-danger" onClick={onConfirm}>
                        Delete
                    </button>
                </div>
            </div>
        </div>
    );
}

/** Three dots keeping time while a question is being worked on. */
function ThinkingDots() {
    return (
        <span className="thinking-dots" aria-hidden="true">
            <span />
            <span />
            <span />
        </span>
    );
}

function MicIcon() {
    return (
        <svg
            viewBox="0 0 24 24"
            width="17"
            height="17"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.9"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
        >
            <rect x="9" y="2.5" width="6" height="11" rx="3" />
            <path d="M5 11a7 7 0 0 0 14 0" />
            <path d="M12 18v3.5" />
        </svg>
    );
}

/** One step of the answer's trail: what we did, why, and what came back. */
function TraceCard({ entry }: { entry: TraceEntry }) {
    const detail = entry.result ?? entry.answer ?? entry.comment;
    const label = TRACE_LABELS[entry.action] ?? entry.action;
    return (
        <div className="trace-entry">
            <div className="trace-action">{label}</div>
            {entry.reason && <div className="trace-reason">{entry.reason}</div>}
            {entry.instruct && (
                <div className="trace-instruct">Looking for: {entry.instruct}</div>
            )}
            {detail && (
                <details>
                    <summary>What came back</summary>
                    <pre>{detail}</pre>
                </details>
            )}
        </div>
    );
}

/** A small waveform that keeps time while there is nothing to play yet. */
function SoundMark() {
    return (
        <div className="soundmark" aria-hidden="true">
            {[0, 1, 2, 3, 4].map((i) => (
                <span key={i} style={{ animationDelay: `${i * 110}ms` }} />
            ))}
        </div>
    );
}

/** Inline so it needs no icon dependency, and inherits the button's color. */
function BackIcon() {
    return (
        <svg
            viewBox="0 0 24 24"
            width="15"
            height="15"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
        >
            <path d="M15 5l-7 7 7 7" />
        </svg>
    );
}

/** Inline so it needs no icon dependency, and inherits the button's color. */
function TrashIcon() {
    return (
        <svg
            viewBox="0 0 24 24"
            width="15"
            height="15"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
        >
            <path d="M3 6h18" />
            <path d="M8 6V4h8v2" />
            <path d="M19 6l-1 14H6L5 6" />
            <path d="M10 11v6M14 11v6" />
        </svg>
    );
}
