import { FileText, Camera, ScanFace, AudioLines, Sparkles, type LucideIcon } from 'lucide-react'

interface Step {
  num: string
  icon: LucideIcon
  label: string
  accent?: boolean
}

const STEPS: Step[] = [
  { num: '01', icon: FileText, label: 'Text' },
  { num: '02', icon: Camera, label: 'Photo' },
  { num: '03', icon: ScanFace, label: 'Live Face' },
  { num: '04', icon: AudioLines, label: 'Voice' },
  { num: '05', icon: Sparkles, label: 'AI Fusion', accent: true },
]

export function HowItWorks() {
  return (
    <section id="how-it-works" className="scroll-mt-20 border-y border-border/70 bg-muted/30">
      <div className="mx-auto w-full max-w-[1216px] px-4 py-16 sm:px-6">
        <div className="mb-10 flex flex-col gap-3">
          <h2 className="text-3xl font-semibold tracking-tight text-balance sm:text-4xl">
            How it works
          </h2>
          <p className="max-w-2xl text-muted-foreground text-pretty">
            You move through four short analysis steps. EmotionDep then fuses the results into
            a single, interpretable screening outcome.
          </p>
        </div>

        <ol className="flex flex-col gap-0 lg:flex-row lg:items-stretch lg:gap-0">
          {STEPS.map((s, i) => (
            <li key={s.num} className="flex flex-1 items-stretch">
              <div className="flex flex-1 flex-col items-center gap-3 py-4 text-center">
                <span
                  className={`flex size-14 items-center justify-center rounded-2xl ring-1 ring-inset ${
                    s.accent
                      ? 'bg-ai/10 text-ai ring-ai/20'
                      : 'bg-card text-foreground/80 ring-border'
                  }`}
                >
                  <s.icon className="size-6" />
                </span>
                <div className="flex flex-col gap-0.5">
                  <span className="font-mono text-xs text-muted-foreground/60">{s.num}</span>
                  <span className="text-sm font-medium">{s.label}</span>
                </div>
              </div>
              {i < STEPS.length - 1 ? (
                <div
                  className="mx-2 hidden w-full max-w-16 self-center border-t border-dashed border-border lg:block"
                  aria-hidden
                />
              ) : null}
            </li>
          ))}
        </ol>
      </div>
    </section>
  )
}
