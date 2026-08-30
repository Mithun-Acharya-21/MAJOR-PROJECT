import { FileText, Camera, ScanFace, AudioLines, Layers } from 'lucide-react'

const SIGNALS = [
  { icon: FileText, label: 'Text' },
  { icon: Camera, label: 'Photo' },
  { icon: ScanFace, label: 'Face' },
  { icon: AudioLines, label: 'Voice' },
]

export function ResearchSection() {
  return (
    <section id="research" className="scroll-mt-20">
      <div className="mx-auto grid w-full max-w-[1216px] items-center gap-12 px-4 py-16 sm:px-6 lg:grid-cols-2">
        <div className="flex flex-col gap-4">
          <span className="text-sm font-medium text-ai">Why multimodal?</span>
          <h2 className="text-3xl font-semibold tracking-tight text-balance sm:text-4xl">
            Different signals offer different perspectives
          </h2>
          <p className="text-muted-foreground text-pretty leading-relaxed">
            NeuroScreen combines multiple input modalities rather than relying on a single
            signal. Language, facial expression, and speech each capture something distinct, and
            weighing them together produces a more balanced screening perspective than any one
            alone.
          </p>
          <p className="text-sm text-muted-foreground/80 text-pretty">
            Combining modalities broadens the evidence considered — it does not guarantee
            medical accuracy.
          </p>
        </div>

        <div className="rounded-3xl border border-border bg-card p-8">
          <div className="flex flex-wrap items-center justify-center gap-3">
            {SIGNALS.map((s) => (
              <div
                key={s.label}
                className="flex items-center gap-2 rounded-xl border border-border bg-background px-3 py-2"
              >
                <s.icon className="size-4 text-foreground/70" />
                <span className="text-sm font-medium">{s.label}</span>
              </div>
            ))}
          </div>
          <div className="my-6 flex justify-center">
            <div className="h-8 w-px bg-gradient-to-b from-border to-ai/40" aria-hidden />
          </div>
          <div className="flex justify-center">
            <div className="inline-flex items-center gap-2.5 rounded-2xl border border-ai/20 bg-ai/10 px-5 py-3 text-ai">
              <Layers className="size-5" />
              <span className="font-medium">Multimodal evidence</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
