/**
 * API client for the EmotionDep FastAPI backend.
 */

const API_BASE = "http://localhost:8000";

// --- Types ---

export type ModalityResult = {
  prediction: string;
  probability: number;
  confidence: number;
  probabilities: { depressed: number; non_depressed: number };
  features: Record<string, unknown>;
  explanation: unknown;
  risk_level: string;
  recommendations?: unknown;
  error?: string;
  warning?: string;
};

export type TextResult = ModalityResult & {
  transcript?: string;
};

export type PhotoResult = ModalityResult & {
  explanation: {
    method: string;
    face_detected: boolean;
    gradcam?: string;
    heatmap?: string;
  };
};

export type VoiceResult = ModalityResult & {
  transcript?: string;
  voice_features?: Record<string, number>;
};

export type MultimodalResult = {
  prediction: string;
  probability: number;
  confidence: number;
  probabilities: { depressed: number; non_depressed: number };
  individual_results: Record<string, ModalityResult>;
  fusion_weights: Record<string, number>;
  explanation: {
    method: string;
    contributions: Array<{
      modality: string;
      probability: number;
      weight: number;
      contribution: number;
      level: string;
    }>;
    available_modalities: string[];
    agreement_score: number;
  };
  risk_level: string;
  recommendations?: unknown;
};

export type AnalysisResult = {
  prediction: string;
  confidence: number;
  probabilities: { depressed: number; non_depressed: number };
  modalities_used: string[];
  text_explanation: Array<{ token: string; weight: number; color?: string }> | null;
  image_explanation: {
    original?: string;
    gradcam?: string;
    overlay?: string;
  } | null;
  risk_level: string;
  transcript?: string;
  session_id?: string;
  voice_features?: Record<string, number> | null;
  voice_score?: number | null;
  num_segments?: number;
  modalities_available?: Record<string, boolean>;
  errors?: string[];
  recommendations?: {
    status: string;
    headline?: string;
    immediate_actions?: string[];
    professional_referrals?: string[];
    coping_strategies?: string[];
    safety_concern?: boolean;
    safety_message?: string;
    what_you_can_do?: {
      today?: string[];
      this_week?: string[];
      when_to_seek_help?: string[];
    };
    explainable_signals?: Array<{ signal: string; description: string; recommendation: string }>;
  };
  risk_status?: string;
};

// --- Legacy endpoint (kept for backward compatibility) ---

export async function analyzeInput(
  text: string | null,
  imageFile: File | null,
): Promise<AnalysisResult> {
  const formData = new FormData();
  if (text) formData.append("text", text);
  if (imageFile) formData.append("file", imageFile);

  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `Analysis failed (${res.status})`);
  }

  return res.json();
}

// --- Step-by-step modality endpoints ---

export async function analyzeText(text: string): Promise<TextResult> {
  const formData = new FormData();
  formData.append("text", text);

  const res = await fetch(`${API_BASE}/api/analyze/text`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `Text analysis failed (${res.status})`);
  }

  return res.json();
}

export async function analyzePhoto(file: File): Promise<PhotoResult> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/api/analyze/photo`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `Photo analysis failed (${res.status})`);
  }

  return res.json();
}

export async function analyzeVoice(
  audioBlob: Blob,
  sessionId?: string,
): Promise<VoiceResult> {
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.webm");
  if (sessionId) formData.append("session_id", sessionId);

  const res = await fetch(`${API_BASE}/api/analyze/voice`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `Voice analysis failed (${res.status})`);
  }

  return res.json();
}

export async function analyzeMultimodal(params: {
  text?: string;
  photo?: File;
  audio?: Blob;
  session_id?: string;
}): Promise<MultimodalResult> {
  const formData = new FormData();
  if (params.text) formData.append("text", params.text);
  if (params.photo) formData.append("photo", params.photo);
  if (params.audio) formData.append("audio", params.audio, "recording.webm");
  if (params.session_id) formData.append("session_id", params.session_id);

  const res = await fetch(`${API_BASE}/api/analyze/multimodal`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `Multimodal analysis failed (${res.status})`);
  }

  return res.json();
}

export async function startSession(): Promise<{ session_id: string; status: string }> {
  const res = await fetch(`${API_BASE}/session/start`, { method: "POST" });
  return res.json();
}

export async function checkHealth(): Promise<{
  status: string;
  models_loaded: boolean;
  device: string;
}> {
  const res = await fetch(`${API_BASE}/api/health`);
  return res.json();
}

// --- Baselines / Model Performance ---

export type ModelMetric = {
  name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  auc: number;
  type: string;
  description?: string;
};

export type BaselinesResult = {
  models: ModelMetric[];
  description?: string;
  datasets?: Record<string, string[]>;
  notes?: string;
};

export async function fetchBaselines(): Promise<BaselinesResult> {
  const res = await fetch(`${API_BASE}/api/baselines`);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `Failed to fetch baselines (${res.status})`);
  }
  return res.json();
}

export async function checkCORS(): Promise<{ ok: boolean }> {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { mode: 'cors' });
    return { ok: res.ok };
  } catch {
    return { ok: false };
  }
}
