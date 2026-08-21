/**
 * Browser microphone recording hook.
 * Always requires explicit user action — never auto-starts.
 * Returns a WAV Blob after recording stops.
 */

import { useState, useRef, useCallback } from "react";

type RecorderState = "idle" | "recording" | "stopped" | "error";

interface UseAudioRecorderResult {
  state: RecorderState;
  audioBlob: Blob | null;
  error: string | null;
  start: () => Promise<void>;
  stop: () => void;
  reset: () => void;
}

export function useAudioRecorder(): UseAudioRecorderResult {
  const [state, setState] = useState<RecorderState>("idle");
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const start = useCallback(async () => {
    try {
      setError(null);
      setAudioBlob(null);
      chunksRef.current = [];
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";
      const recorder = new MediaRecorder(stream, { mimeType });
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType });
        setAudioBlob(blob);
        setState("stopped");
        stream.getTracks().forEach((t) => t.stop());
      };
      recorder.onerror = () => {
        setError("Recording error. Please try again.");
        setState("error");
      };
      recorder.start(250);
      mediaRef.current = recorder;
      setState("recording");
    } catch (err) {
      setError(
        "Microphone access denied or unavailable. " +
        "Please check your browser permissions."
      );
      setState("error");
    }
  }, []);

  const stop = useCallback(() => {
    if (mediaRef.current && mediaRef.current.state === "recording") {
      mediaRef.current.stop();
    }
  }, []);

  const reset = useCallback(() => {
    setAudioBlob(null);
    setError(null);
    setState("idle");
    chunksRef.current = [];
  }, []);

  return { state, audioBlob, error, start, stop, reset };
}
