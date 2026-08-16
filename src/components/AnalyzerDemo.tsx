import { useMemo, useState } from "react";
import { Brain, ScanFace, Sparkles, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { runMockInference, type TokenWeight } from "@/lib/mock-inference";

const MOODS = [
  { id: "none", label: "No image" },
  { id: "neutral", label: "Neutral face" },
  { id: "sad", label: "Low-affect face" },
  { id: "happy", label: "Positive face" },
] as const;

type MoodId = (typeof MOODS)[number]["id"];

const SAMPLES = [
  "Been awake since 3am again. Everything feels heavy and pointless, I'm just so tired of pretending I'm fine.",
  "Rough week but I laughed with friends today and slept better. Things feel a little more hopeful.",
];

function tokenStyle(weight: number) {
  if (weight > 0.05)
    return {
      backgroundColor: `color-mix(in oklch, var(--risk) ${Math.round(weight * 55)}%, transparent)`,
    };
  if (weight < -0.05)
    return {
      backgroundColor: `color-mix(in oklch, var(--calm) ${Math.round(Math.abs(weight) * 55)}%, transparent)`,
    };
  return undefined;
}

function LimeText({ tokens }: { tokens: TokenWeight[] }) {
  return (
    <p className="text-sm leading-7 text-foreground/90">
      {tokens.map((t, i) =>
        t.token.trim() === "" ? (
          <span key={i}>{t.token}</span>
        ) : (
          <span key={i} className="rounded px-0.5 py-0.5" style={tokenStyle(t.weight)}>
            {t.token}
          </span>
        ),
      )}
    </p>
  );
}

export function AnalyzerDemo() {
  const [text, setText] = useState(SAMPLES[0]!);
  const [mood, setMood] = useState<MoodId>("sad");
  const [submitted, setSubmitted] = useState<{ text: string; mood: MoodId } | null>(null);

  const result = useMemo(
    () => (submitted ? runMockInference(submitted.text, submitted.mood) : null),
    [submitted],
  );

  const riskColor =
    result && result.probability >= 0.66
      ? "var(--risk)"
      : result && result.probability >= 0.38
        ? "var(--attention)"
        : "var(--calm)";

  return (
    <div className="grid gap-6 lg:grid-cols-[1.05fr_1fr]">
      <div className="panel p-6">
        <div className="flex items-center gap-2">
          <Brain className="size-4 text-primary" />
          <h3 className="text-lg font-semibold">Multimodal input</h3>
        </div>
        <p className="mt-1 text-sm text-muted-foreground">
          Type a short post and pick a facial-affect sample. Runs a simulated forward pass — no data
          leaves your browser.
        </p>

        <label className="mt-5 block text-xs font-medium uppercase tracking-widest text-muted-foreground">
          Text branch input
        </label>
        <Textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={5}
          className="mt-2 resize-none bg-input/40 font-mono text-sm"
          placeholder="Paste a social media post or journal entry…"
        />
        <div className="mt-2 flex flex-wrap gap-2">
          {SAMPLES.map((s, i) => (
            <Button key={i} variant="secondary" size="sm" onClick={() => setText(s)}>
              Sample {i + 1}
            </Button>
          ))}
        </div>

        <label className="mt-6 block text-xs font-medium uppercase tracking-widest text-muted-foreground">
          Facial branch input
        </label>
        <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
          {MOODS.map((m) => (
            <button
              key={m.id}
              onClick={() => setMood(m.id)}
              className={`rounded-lg border px-3 py-3 text-xs font-medium transition-colors ${
                mood === m.id
                  ? "border-primary bg-primary/15 text-primary"
                  : "border-border bg-secondary/40 text-muted-foreground hover:bg-secondary"
              }`}
            >
              <ScanFace className="mx-auto mb-1 size-4" />
              {m.label}
            </button>
          ))}
        </div>

        <div className="mt-6 flex gap-2">
          <Button
            className="flex-1"
            onClick={() => setSubmitted({ text, mood })}
            disabled={!text.trim()}
          >
            <Sparkles className="size-4" /> Run inference
          </Button>
          <Button variant="outline" onClick={() => setSubmitted(null)}>
            <RotateCcw className="size-4" /> Reset
          </Button>
        </div>
      </div>

      <div className="panel p-6">
        <h3 className="text-lg font-semibold">Prediction &amp; explanation</h3>
        {!result ? (
          <div className="mt-10 flex flex-col items-center justify-center gap-3 text-center text-sm text-muted-foreground">
            <Sparkles className="size-6 text-primary/60" />
            Run inference to see the fused score, per-branch confidence, LIME token attribution and
            emotion distribution.
          </div>
        ) : (
          <div className="mt-4 space-y-6">
            <div className="rounded-xl border border-border bg-background/40 p-4">
              <div className="flex items-end justify-between">
                <div>
                  <p className="text-xs uppercase tracking-widest text-muted-foreground">
                    Fusion output
                  </p>
                  <p className="mt-1 text-3xl font-semibold" style={{ color: riskColor }}>
                    {(result.probability * 100).toFixed(1)}%
                  </p>
                </div>
                <Badge variant="outline" style={{ color: riskColor, borderColor: riskColor }}>
                  {result.label}
                </Badge>
              </div>
              <Progress value={result.probability * 100} className="mt-3 h-2" />
            </div>

            <div className="grid grid-cols-2 gap-3">
              {result.contributions.map((c) => (
                <div key={c.modality} className="rounded-lg border border-border bg-background/30 p-3">
                  <p className="text-[11px] leading-tight text-muted-foreground">{c.modality}</p>
                  <p className="mt-2 font-mono text-lg">{c.value.toFixed(3)}</p>
                </div>
              ))}
            </div>

            <div>
              <p className="text-xs uppercase tracking-widest text-muted-foreground">
                LIME token attribution
              </p>
              <div className="mt-2 rounded-lg border border-border bg-background/30 p-3">
                <LimeText tokens={result.tokens} />
              </div>
              <div className="mt-2 flex gap-4 text-[11px] text-muted-foreground">
                <span className="flex items-center gap-1">
                  <span className="size-2 rounded-full" style={{ background: "var(--risk)" }} />
                  pushes toward depression
                </span>
                <span className="flex items-center gap-1">
                  <span className="size-2 rounded-full" style={{ background: "var(--calm)" }} />
                  pushes away
                </span>
              </div>
            </div>

            <div>
              <p className="text-xs uppercase tracking-widest text-muted-foreground">
                Facial emotion distribution
              </p>
              <div className="mt-3 space-y-2">
                {result.emotions.map((e) => (
                  <div key={e.label} className="flex items-center gap-3">
                    <span className="w-16 text-xs text-muted-foreground">{e.label}</span>
                    <div className="h-2 flex-1 overflow-hidden rounded-full bg-secondary">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${e.value * 100}%`,
                          background: "var(--gradient-signal)",
                        }}
                      />
                    </div>
                    <span className="w-10 text-right font-mono text-xs">
                      {(e.value * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
