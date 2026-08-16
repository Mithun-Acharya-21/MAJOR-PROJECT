export type TokenWeight = { token: string; weight: number };

export type EmotionScore = { label: string; value: number };

export type InferenceResult = {
  probability: number;
  label: "Low indication" | "Moderate indication" | "High indication";
  textScore: number;
  faceScore: number;
  tokens: TokenWeight[];
  emotions: EmotionScore[];
  contributions: { modality: string; value: number }[];
};

const NEGATIVE = [
  "hopeless","worthless","empty","tired","exhausted","alone","lonely","numb","sad","cry",
  "crying","insomnia","sleep","nothing","pointless","hate","failure","burden","give","up",
  "cant","can't","anymore","dark","heavy","broken","lost","guilt","ashamed","suicidal",
];
const POSITIVE = [
  "happy","grateful","hopeful","better","excited","love","enjoy","progress","calm","proud",
  "rested","friends","future","good","great","laughed","improving",
];

const EMOTION_PRESETS: Record<string, EmotionScore[]> = {
  neutral: [
    { label: "Neutral", value: 0.62 },
    { label: "Sad", value: 0.14 },
    { label: "Fear", value: 0.08 },
    { label: "Happy", value: 0.1 },
    { label: "Anger", value: 0.06 },
  ],
  sad: [
    { label: "Sad", value: 0.68 },
    { label: "Neutral", value: 0.16 },
    { label: "Fear", value: 0.09 },
    { label: "Anger", value: 0.05 },
    { label: "Happy", value: 0.02 },
  ],
  happy: [
    { label: "Happy", value: 0.71 },
    { label: "Neutral", value: 0.18 },
    { label: "Sad", value: 0.05 },
    { label: "Fear", value: 0.03 },
    { label: "Anger", value: 0.03 },
  ],
  none: [
    { label: "Neutral", value: 0.5 },
    { label: "Sad", value: 0.2 },
    { label: "Happy", value: 0.15 },
    { label: "Fear", value: 0.09 },
    { label: "Anger", value: 0.06 },
  ],
};

const clamp = (n: number) => Math.min(0.98, Math.max(0.02, n));

export function runMockInference(text: string, faceMood: keyof typeof EMOTION_PRESETS): InferenceResult {
  const rawTokens = text.split(/(\s+)/).filter((t) => t.length > 0);
  const tokens: TokenWeight[] = rawTokens.map((token) => {
    const clean = token.toLowerCase().replace(/[^a-z']/g, "");
    let weight = 0;
    if (clean.length > 2 && NEGATIVE.includes(clean)) weight = 0.45 + (clean.length % 5) * 0.1;
    else if (clean.length > 2 && POSITIVE.includes(clean)) weight = -(0.4 + (clean.length % 4) * 0.1);
    return { token, weight: Math.max(-1, Math.min(1, weight)) };
  });

  const positives = tokens.filter((t) => t.weight > 0).reduce((a, t) => a + t.weight, 0);
  const negatives = tokens.filter((t) => t.weight < 0).reduce((a, t) => a + Math.abs(t.weight), 0);
  const words = Math.max(1, text.trim().split(/\s+/).filter(Boolean).length);

  const textScore = clamp(0.2 + (positives - negatives * 0.9) / Math.sqrt(words) / 1.6);

  const emotions: EmotionScore[] = EMOTION_PRESETS[faceMood] ?? EMOTION_PRESETS["none"]!;
  const sad = emotions.find((e) => e.label === "Sad")?.value ?? 0.2;
  const happy = emotions.find((e) => e.label === "Happy")?.value ?? 0.2;
  const faceScore = faceMood === "none" ? textScore : clamp(0.25 + sad * 0.95 - happy * 0.5);

  const wText = faceMood === "none" ? 1 : 0.62;
  const probability = clamp(textScore * wText + faceScore * (1 - wText));

  const label =
    probability >= 0.66 ? "High indication" : probability >= 0.38 ? "Moderate indication" : "Low indication";

  return {
    probability,
    label,
    textScore,
    faceScore,
    tokens,
    emotions,
    contributions: [
      { modality: "Linguistic (DeBERTa-v3 + BiLSTM-Attn)", value: textScore * wText },
      { modality: "Facial affect (CNN)", value: faceScore * (1 - wText) },
    ],
  };
}
