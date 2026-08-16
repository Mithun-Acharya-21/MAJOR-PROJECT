import { createFileRoute } from "@tanstack/react-router";
import {
  Activity,
  FileText,
  ScanFace,
  Layers,
  Lightbulb,
  BarChart3,
  FolderTree,
  ShieldAlert,
} from "lucide-react";
import { AnalyzerDemo } from "@/components/AnalyzerDemo";
import { Badge } from "@/components/ui/badge";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Emotion-Aware Depression Detection | Explainable Multimodal AI" },
      {
        name: "description",
        content:
          "Interactive console for an explainable deep learning system that detects early depression signals from text (DeBERTa-v3 + BiLSTM-Attention) and facial emotion (CNN), fused with LIME/SHAP explanations.",
      },
      { property: "og:title", content: "Emotion-Aware Depression Detection" },
      {
        property: "og:description",
        content:
          "Explainable multimodal deep learning: text and facial-emotion branches fused for early depression detection, with LIME and SHAP explanations.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

const BRANCHES = [
  {
    icon: FileText,
    title: "Text pipeline",
    color: "var(--text-branch)",
    items: ["Slang & emoji normalisation", "DeBERTa-v3 encoder", "BiLSTM + Attention head"],
    owner: "Members 1 & 2",
  },
  {
    icon: ScanFace,
    title: "Facial pipeline",
    color: "var(--face-branch)",
    items: ["Face detection & cropping", "FER-2013 / RAF-DB training", "CNN emotion classifier"],
    owner: "Member 3",
  },
  {
    icon: Layers,
    title: "Fusion & XAI",
    color: "var(--fusion-branch)",
    items: ["Gated feature fusion", "Classifier head", "LIME + SHAP explainers"],
    owner: "Member 4",
  },
];

const STRUCTURE = [
  { dir: "data/", note: "raw · processed · external dictionaries" },
  { dir: "notebooks/", note: "01→06 exploration to explainability" },
  { dir: "src/text_pipeline/", note: "preprocessing, tokenizer, DeBERTa, BiLSTM-Attn" },
  { dir: "src/face_pipeline/", note: "detection, preprocessing, CNN, training" },
  { dir: "src/fusion/", note: "feature fusion, classifier head, training" },
  { dir: "src/explainability/", note: "LIME, SHAP, visualisation" },
  { dir: "src/evaluation/", note: "metrics, baselines (NB, SVM, BERT)" },
  { dir: "configs/", note: "text / face / fusion / training YAML" },
  { dir: "models/", note: "checkpoints + final best_model.pt" },
  { dir: "results/", note: "metrics, ROC, confusion, XAI outputs" },
  { dir: "app/", note: "demo interface + inference" },
  { dir: "tests/", note: "preprocessing, models, fusion" },
];

const METRICS = [
  { label: "Text only (F1)", value: "0.86", tone: "var(--text-branch)" },
  { label: "Face only (F1)", value: "0.79", tone: "var(--face-branch)" },
  { label: "Fusion (F1)", value: "0.91", tone: "var(--fusion-branch)" },
  { label: "Fusion ROC-AUC", value: "0.94", tone: "var(--calm)" },
];

function Index() {
  return (
    <main className="min-h-screen">
      <section className="hero-surface relative overflow-hidden border-b border-border">
        <div className="grid-surface absolute inset-0 opacity-70" aria-hidden />
        <div className="relative mx-auto max-w-6xl px-6 py-20">
          <Badge variant="outline" className="border-primary/40 text-primary">
            <Activity className="size-3" /> Phase I · Research console
          </Badge>
          <h1 className="mt-6 max-w-3xl text-4xl font-semibold leading-tight sm:text-5xl">
            Emotion-aware, <span className="text-gradient-signal">explainable</span> deep learning
            for early depression detection
          </h1>
          <p className="mt-5 max-w-2xl text-base leading-relaxed text-muted-foreground">
            A multimodal system that reads what people write and how they look — a DeBERTa-v3 +
            BiLSTM-Attention text branch and a CNN facial-emotion branch, fused into one score that
            always explains itself with LIME and SHAP.
          </p>
          <div className="mt-8 grid gap-4 sm:grid-cols-4">
            {METRICS.map((m) => (
              <div key={m.label} className="panel p-4">
                <p className="font-mono text-2xl" style={{ color: m.tone }}>
                  {m.value}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">{m.label}</p>
              </div>
            ))}
          </div>
          <p className="mt-3 text-[11px] text-muted-foreground">
            Target figures for Phase II evaluation — placeholders until training runs land in{" "}
            <span className="font-mono">results/metrics/</span>.
          </p>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-16">
        <div className="mb-6 flex items-center gap-2">
          <Lightbulb className="size-4 text-primary" />
          <h2 className="text-2xl font-semibold">Live demo</h2>
        </div>
        <AnalyzerDemo />
      </section>

      <section className="border-y border-border bg-surface/40">
        <div className="mx-auto max-w-6xl px-6 py-16">
          <h2 className="text-2xl font-semibold">Architecture branches</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            Three independent branches, one fused classifier head — mapped to how the team of four
            splits the repo.
          </p>
          <div className="mt-8 grid gap-5 md:grid-cols-3">
            {BRANCHES.map((b) => (
              <div key={b.title} className="panel p-6">
                <b.icon className="size-5" style={{ color: b.color }} />
                <h3 className="mt-4 text-lg font-semibold">{b.title}</h3>
                <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
                  {b.items.map((i) => (
                    <li key={i} className="flex gap-2">
                      <span
                        className="mt-1.5 size-1.5 shrink-0 rounded-full"
                        style={{ background: b.color }}
                      />
                      {i}
                    </li>
                  ))}
                </ul>
                <p className="mt-5 text-xs uppercase tracking-widest text-muted-foreground">
                  {b.owner}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-16">
        <div className="grid gap-10 lg:grid-cols-[1.1fr_1fr]">
          <div>
            <div className="flex items-center gap-2">
              <FolderTree className="size-4 text-primary" />
              <h2 className="text-2xl font-semibold">Repository map</h2>
            </div>
            <p className="mt-2 text-sm text-muted-foreground">
              Pipelines, configs, artefacts and reports stay cleanly separated so four people can
              commit without merge conflicts.
            </p>
            <div className="panel mt-6 divide-y divide-border">
              {STRUCTURE.map((s) => (
                <div key={s.dir} className="flex items-baseline justify-between gap-4 px-4 py-2.5">
                  <span className="font-mono text-sm text-primary">{s.dir}</span>
                  <span className="text-right text-xs text-muted-foreground">{s.note}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="space-y-5">
            <div className="panel p-6">
              <BarChart3 className="size-5 text-primary" />
              <h3 className="mt-4 text-lg font-semibold">Evaluation plan</h3>
              <p className="mt-2 text-sm text-muted-foreground">
                Baselines (Naïve Bayes, SVM, plain BERT) versus text-only, face-only and fused
                models — accuracy, precision, recall, F1 and ROC-AUC written to{" "}
                <span className="font-mono">results/metrics/</span> as JSON for report tables.
              </p>
            </div>
            <div className="panel p-6">
              <ShieldAlert className="size-5" style={{ color: "var(--attention)" }} />
              <h3 className="mt-4 text-lg font-semibold">Ethics &amp; limitations</h3>
              <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
                <li>Screening research prototype — not a diagnosis, not a clinical device.</li>
                <li>Sarcasm, masking and posed expressions degrade both branches.</li>
                <li>Dataset bias across demographics is documented in docs/limitations.md.</li>
                <li>No personal data is stored by this demo interface.</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <footer className="border-t border-border">
        <div className="mx-auto max-w-6xl px-6 py-10 text-xs text-muted-foreground">
          Emotion-Aware &amp; Explainable Deep Learning Model for Early Depression Detection ·
          Phase I academic project. If you or someone you know is struggling, contact a local mental
          health helpline.
        </div>
      </footer>
    </main>
  );
}
