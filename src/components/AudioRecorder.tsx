import { useRef, useState, useEffect } from "react";
import { Mic, Square, Loader2, Play } from "lucide-react";
import { Button } from "@/components/ui/button";

interface AudioRecorderProps {
  onRecord: (blob: Blob) => void;
  onAnalyze: () => void;
  loading: boolean;
  hasRecording: boolean;
}

export function AudioRecorder({
  onRecord,
  onAnalyze,
  loading,
  hasRecording,
}: AudioRecorderProps) {
  const [recording, setRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm;codecs=opus",
      });
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        onRecord(blob);
        stream.getTracks().forEach((t) => t.stop());
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start(100);
      setRecording(true);
      setDuration(0);
      setError(null);

      timerRef.current = setInterval(() => {
        setDuration((d) => d + 1);
      }, 1000);
    } catch {
      setError("Microphone access denied. Please allow microphone access in your browser.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    setRecording(false);
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const formatDuration = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  return (
    <div className="mt-4">
      <div className="rounded-lg border border-border bg-secondary/20 p-4">
        <div className="flex items-center gap-4">
          <div
            className={`flex size-16 items-center justify-center rounded-full ${
              recording
                ? "bg-red-500/20 text-red-500 animate-pulse"
                : "bg-primary/10 text-primary"
            }`}
          >
            {recording ? (
              <Mic className="size-6" />
            ) : (
              <Mic className="size-6" />
            )}
          </div>
          <div>
            {recording ? (
              <>
                <p className="text-sm font-medium">Recording...</p>
                <p className="font-mono text-lg text-red-500">
                  {formatDuration(duration)}
                </p>
              </>
            ) : hasRecording ? (
              <>
                <p className="text-sm font-medium text-green-500">Recording saved</p>
                <p className="text-xs text-muted-foreground">Duration: {formatDuration(duration)}</p>
              </>
            ) : (
              <>
                <p className="text-sm font-medium">Ready to record</p>
                <p className="text-xs text-muted-foreground">
                  Click to start recording your voice
                </p>
              </>
            )}
          </div>
        </div>

        {error && (
          <p className="mt-3 text-sm text-destructive">{error}</p>
        )}

        <div className="mt-4 flex gap-2">
          {!recording ? (
            <Button onClick={startRecording} disabled={loading}>
              <Mic className="size-4" /> Start Recording
            </Button>
          ) : (
            <Button onClick={stopRecording} variant="destructive">
              <Square className="size-4" /> Stop
            </Button>
          )}
          {hasRecording && !recording && (
            <Button onClick={onAnalyze} disabled={loading}>
              {loading ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Play className="size-4" />
              )}
              Analyze Voice
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
