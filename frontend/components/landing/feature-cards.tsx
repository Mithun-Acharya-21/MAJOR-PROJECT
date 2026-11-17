import { FileText, Camera, ScanFace, AudioLines, ArrowUpRight, type LucideIcon } from 'lucide-react'

interface Feature {
  num: string
  icon: LucideIcon
  title: string
  desc: string
}

const FEATURES: Feature[] = [
  {
    num: '01',
    icon: FileText,
    title: 'Text Analysis',
    desc: 'Analyze written responses for linguistic patterns associated with mood and affect.',
  },
  {
    num: '02',
    icon: Camera,
    title: 'Photo Analysis',
    desc: 'Analyze facial-expression information from a single uploaded image.',
  },
  {
    num: '03',
    icon: ScanFace,
    title: 'Live Face',
    desc: 'Analyze facial expressions across multiple webcam frames in real time.',
  },
  {
    num: '04',
    icon: AudioLines,
    title: 'Voice & Speech',
    desc: 'Analyze speech characteristics alongside the transcribed content.',
  },
]

export function FeatureCards() {
  return (
    <section className="mx-auto w-full max-w-[1216px] px-4 py-16 sm:px-6">
      <div className="mb-10 flex flex-col gap-3">
        <h2 className="text-3xl font-semibold tracking-tight text-balance sm:text-4xl">
          Four signals, one screening
        </h2>
        <p className="max-w-2xl text-muted-foreground text-pretty">
          Each modality contributes a different perspective. EmotionDep analyzes them
          independently, then combines the evidence.
        </p>
      </div>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {FEATURES.map((f) => (
          <article
            key={f.num}
            className="group relative flex flex-col gap-4 rounded-2xl border border-border bg-card p-6 transition-all duration-300 hover:-translate-y-0.5 hover:border-primary/30 hover:shadow-lg hover:shadow-primary/5"
          >
            <div className="flex items-center justify-between">
              <span className="flex size-11 items-center justify-center rounded-xl bg-primary/10 text-primary ring-1 ring-inset ring-primary/15">
                <f.icon className="size-5" />
              </span>
              <span className="font-mono text-sm text-muted-foreground/60">{f.num}</span>
            </div>
            <div className="flex flex-col gap-1.5">
              <h3 className="text-lg font-medium">{f.title}</h3>
              <p className="text-sm leading-relaxed text-muted-foreground text-pretty">
                {f.desc}
              </p>
            </div>
            <ArrowUpRight className="mt-auto size-4 text-muted-foreground/50 transition-colors group-hover:text-primary" />
          </article>
        ))}
      </div>
    </section>
  )
}
