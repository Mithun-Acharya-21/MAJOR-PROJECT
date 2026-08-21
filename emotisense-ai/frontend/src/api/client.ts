/**
 * Typed API client for the EmotiSense AI backend.
 *
 * All calls go through /api/v1 which Vite proxies to http://localhost:8000
 * in development. In production the same origin serves both.
 */

import type {
  ConsentResponse,
  HealthResponse,
  ImageAnalysisResponse,
  ModelInfoResponse,
  MultimodalAnalysisResponse,
  TextAnalysisResponse,
  VideoAnalysisResponse,
  VoiceAnalysisResponse,
} from "@/types/api";

const BASE = "/api/v1";

async function request<T>(
  path: string,
  init: RequestInit = {}
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { Accept: "application/json", ...init.headers },
    ...init,
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch { /* ignore */ }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

// ---- System ----

export async function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  return request<ModelInfoResponse>("/model/info");
}

// ---- Consent ----

export async function recordConsent(
  consentType: string
): Promise<ConsentResponse> {
  return request<ConsentResponse>("/analyze/consent", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ consent_type: consentType, user_acknowledged: true }),
  });
}

// ---- Text ----

export async function analyzeText(
  text: string
): Promise<TextAnalysisResponse> {
  return request<TextAnalysisResponse>("/analyze/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, consent_given: true }),
  });
}

// ---- Voice ----

export async function analyzeVoice(
  audioFile: File
): Promise<VoiceAnalysisResponse> {
  const fd = new FormData();
  fd.append("audio", audioFile);
  fd.append("consent_given", "true");
  return request<VoiceAnalysisResponse>("/analyze/voice", {
    method: "POST",
    body: fd,
  });
}

// ---- Image ----

export async function analyzeImage(
  imageFile: File
): Promise<ImageAnalysisResponse> {
  const fd = new FormData();
  fd.append("image", imageFile);
  fd.append("consent_given", "true");
  return request<ImageAnalysisResponse>("/analyze/image", {
    method: "POST",
    body: fd,
  });
}

// ---- Video ----

export async function analyzeVideo(
  videoFile: File
): Promise<VideoAnalysisResponse> {
  const fd = new FormData();
  fd.append("video", videoFile);
  fd.append("consent_given", "true");
  return request<VideoAnalysisResponse>("/analyze/video", {
    method: "POST",
    body: fd,
  });
}

// ---- Multimodal ----

export async function analyzeMultimodal(opts: {
  text?: string;
  audioFile?: File;
  imageFile?: File;
}): Promise<MultimodalAnalysisResponse> {
  const fd = new FormData();
  fd.append("consent_given", "true");
  if (opts.text) fd.append("text", opts.text);
  if (opts.audioFile) fd.append("audio", opts.audioFile);
  if (opts.imageFile) fd.append("image", opts.imageFile);
  return request<MultimodalAnalysisResponse>("/analyze/multimodal", {
    method: "POST",
    body: fd,
  });
}
