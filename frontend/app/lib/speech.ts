"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Speech input for the Q&A box, on top of the browser's Web Speech API.
 *
 * Recognition runs in the browser, not on our workers: the pipeline's Whisper
 * lives behind Celery on the media queue, where a round trip costs seconds and
 * a whole job's worth of infrastructure to transcribe one spoken sentence.
 * The trade is browser support — Chrome, Edge and Safari have it, Firefox does
 * not — so `supported` is false there and callers hide the mic rather than
 * offering a button that cannot work.
 *
 * The hook only reports what it heard; it never decides what a phrase means.
 * That is the seam for wake-word activation later: run a second instance with
 * `continuous: true` that stays listening and matches `onFinal` text against a
 * trigger phrase, then hand the remainder of the utterance to `onAsk`.
 */

interface RecognitionAlternative {
    transcript: string;
}

interface RecognitionResult {
    readonly length: number;
    readonly isFinal: boolean;
    item(index: number): RecognitionAlternative;
    [index: number]: RecognitionAlternative;
}

interface RecognitionResultList {
    readonly length: number;
    item(index: number): RecognitionResult;
    [index: number]: RecognitionResult;
}

interface RecognitionEventLike extends Event {
    readonly resultIndex: number;
    readonly results: RecognitionResultList;
}

interface RecognitionErrorEventLike extends Event {
    readonly error: string;
}

interface RecognitionLike {
    lang: string;
    continuous: boolean;
    interimResults: boolean;
    maxAlternatives: number;
    start(): void;
    stop(): void;
    abort(): void;
    onresult: ((event: RecognitionEventLike) => void) | null;
    onerror: ((event: RecognitionErrorEventLike) => void) | null;
    onend: (() => void) | null;
    onstart: (() => void) | null;
}

type RecognitionCtor = new () => RecognitionLike;

function recognitionCtor(): RecognitionCtor | null {
    if (typeof window === "undefined") return null;
    const w = window as unknown as {
        SpeechRecognition?: RecognitionCtor;
        webkitSpeechRecognition?: RecognitionCtor;
    };
    return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

/** What went wrong, said the way the panel should say it. */
function errorMessage(code: string): string | null {
    switch (code) {
        case "no-speech":
        case "aborted":
            // Both are ordinary ends to a turn, not failures worth reporting.
            return null;
        case "not-allowed":
        case "service-not-allowed":
            return "Microphone access is blocked. Allow it in your browser to ask out loud.";
        case "audio-capture":
            return "No microphone found.";
        case "network":
            return "Speech recognition couldn’t reach the network.";
        default:
            return "Couldn’t hear that — try again.";
    }
}

export interface SpeechInputOptions {
    /** Fires once per settled utterance, with the recognized text. */
    onFinal: (transcript: string) => void;
    /** Fires as words firm up mid-utterance, for a live preview. */
    onInterim?: (transcript: string) => void;
    /**
     * Keep listening across utterances instead of stopping after the first.
     * Push-to-talk dictation wants false; a future wake-word listener wants true.
     */
    continuous?: boolean;
    /** BCP-47 tag; defaults to the browser's language. */
    lang?: string;
}

export interface SpeechInput {
    /** False when the browser has no Web Speech API — hide the mic entirely. */
    supported: boolean;
    listening: boolean;
    /** The words heard so far in the current utterance, not yet final. */
    interim: string;
    error: string | null;
    start: () => void;
    stop: () => void;
    toggle: () => void;
}

export function useSpeechInput(options: SpeechInputOptions): SpeechInput {
    const { continuous = false, lang } = options;
    const [supported, setSupported] = useState(false);
    const [listening, setListening] = useState(false);
    const [interim, setInterim] = useState("");
    const [error, setError] = useState<string | null>(null);

    const recognition = useRef<RecognitionLike | null>(null);
    // Callbacks live in a ref so a re-render with a fresh closure doesn't tear
    // down and rebuild recognition mid-sentence.
    const handlers = useRef(options);
    handlers.current = options;

    // The API only exists in the browser, so support is unknown until mount —
    // deciding it during render would disagree with the server's HTML.
    useEffect(() => setSupported(recognitionCtor() !== null), []);

    useEffect(() => {
        const Ctor = recognitionCtor();
        if (!Ctor) return;

        const rec = new Ctor();
        rec.continuous = continuous;
        rec.interimResults = true;
        rec.maxAlternatives = 1;
        rec.lang = lang ?? navigator.language ?? "en-US";

        rec.onstart = () => {
            setListening(true);
            setError(null);
        };
        rec.onresult = (event) => {
            let pending = "";
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const result = event.results[i];
                const text = result[0]?.transcript ?? "";
                if (result.isFinal) {
                    const settled = text.trim();
                    if (settled) handlers.current.onFinal(settled);
                } else {
                    pending += text;
                }
            }
            setInterim(pending);
            if (pending) handlers.current.onInterim?.(pending);
        };
        rec.onerror = (event) => {
            const message = errorMessage(event.error);
            if (message) setError(message);
        };
        rec.onend = () => {
            setListening(false);
            setInterim("");
        };

        recognition.current = rec;
        return () => {
            rec.onresult = rec.onerror = rec.onend = rec.onstart = null;
            rec.abort();
            recognition.current = null;
        };
    }, [continuous, lang]);

    const start = useCallback(() => {
        const rec = recognition.current;
        if (!rec) return;
        setError(null);
        setInterim("");
        try {
            rec.start();
        } catch {
            /* start() throws if it is already running, which is a no-op here */
        }
    }, []);

    const stop = useCallback(() => recognition.current?.stop(), []);

    const toggle = useCallback(() => {
        if (listening) stop();
        else start();
    }, [listening, start, stop]);

    return { supported, listening, interim, error, start, stop, toggle };
}
