import { useRef, useState, useCallback, useEffect } from "react";
import { Video, Camera, Square, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";

interface WebcamCaptureProps {
  onCapture: (frames: Blob[]) => void;
  loading: boolean;
  maxFrames?: number;
  captureIntervalMs?: number;
}

export function WebcamCapture({
  onCapture,
  loading,
  maxFrames = 10,
  captureIntervalMs = 1000,
}: WebcamCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [streaming, setStreaming] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startStream = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 240, facingMode: "user" },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setStreaming(true);
        setError(null);
      }
    } catch {
      setError("Camera access denied. Please allow camera access in your browser.");
    }
  };

  const stopStream = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    setStreaming(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const captureFrame = useCallback((): Blob | null => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return null;

    canvas.width = video.videoWidth || 320;
    canvas.height = video.videoHeight || 240;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return new Promise<Blob | null>((resolve) => {
      canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.8);
    }) as unknown as Blob | null;
  }, []);

  const startCapturing = () => {
    const frames: Blob[] = [];
    setFrameCount(0);
    setCapturing(true);

    intervalRef.current = setInterval(() => {
      const frame = captureFrame();
      if (frame) {
        frames.push(frame);
        setFrameCount(frames.length);
        if (frames.length >= maxFrames) {
          if (intervalRef.current) clearInterval(intervalRef.current);
          setCapturing(false);
          onCapture(frames);
        }
      }
    }, captureIntervalMs);
  };

  const stopCapturing = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setCapturing(false);
  };

  useEffect(() => {
    return () => {
      stopStream();
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  return (
    <div className="mt-4">
      <div className="relative overflow-hidden rounded-lg border border-border bg-black">
        <video
          ref={videoRef}
          className="w-full max-h-64 object-contain"
          muted
          playsInline
        />
        <canvas ref={canvasRef} className="hidden" />
        {!streaming && !error && (
          <div className="absolute inset-0 flex items-center justify-center text-white/60">
            <div className="text-center">
              <Camera className="mx-auto size-8" />
              <p className="mt-2 text-sm">Click "Start Camera" to begin</p>
            </div>
          </div>
        )}
        {capturing && (
          <div className="absolute top-2 right-2 flex items-center gap-1 rounded bg-red-500/80 px-2 py-1 text-xs text-white">
            <span className="size-2 animate-pulse rounded-full bg-white" />
            Recording ({frameCount}/{maxFrames})
          </div>
        )}
      </div>

      {error && (
        <p className="mt-2 text-sm text-destructive">{error}</p>
      )}

      <div className="mt-3 flex gap-2">
        {!streaming ? (
          <Button onClick={startStream} variant="outline">
            <Video className="size-4" /> Start Camera
          </Button>
        ) : (
          <>
            {!capturing ? (
              <Button onClick={startCapturing} disabled={loading}>
                {loading ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <Camera className="size-4" />
                )}
                Capture Frames
              </Button>
            ) : (
              <Button onClick={stopCapturing} variant="destructive">
                <Square className="size-4" /> Stop
              </Button>
            )}
            <Button onClick={stopStream} variant="ghost" size="sm">
              Stop Camera
            </Button>
          </>
        )}
      </div>
    </div>
  );
}
