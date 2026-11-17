import Link from 'next/link'
import { ArrowRight, ShieldCheck } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ModalityVisual } from './modality-visual'

export function Hero() {
  return (
    <section className="relative overflow-hidden">
      <div className="pointer-events-none absolute inset-0 bg-grid opacity-[0.35] [mask-image:radial-gradient(ellipse_at_center,black,transparent_72%)]" />
      <div className="relative mx-auto grid w-full max-w-[1216px] items-center gap-12 px-4 py-16 sm:px-6 lg:grid-cols-[1.05fr_0.95fr] lg:py-24">
        <div className="flex flex-col items-start gap-6">
          <span className="inline-flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1 text-xs font-medium text-muted-foreground">
            <ShieldCheck className="size-3.5 text-primary" />
            Research & educational screening
          </span>
          <div className="flex flex-col gap-4">
            <h1 className="text-3xl font-semibold tracking-tight text-balance sm:text-4xl lg:text-5xl">
              Emotion-Aware and Explainable Deep Learning Model for Early Depression Detection Using Attention-Based Transformer Networks
            </h1>
            <p className="max-w-xl text-base leading-relaxed text-muted-foreground text-pretty sm:text-lg">
              An AI-powered research tool that analyzes text, facial expressions, photos, and
              voice to identify depression-related indicators and provide interpretable
              screening insights.
            </p>
          </div>
          <div className="flex flex-col gap-3 sm:flex-row">
            <Button
              size="lg"
              className="rounded-xl"
              nativeButton={false}
              render={
                <Link href="/screening">
                  Start screening
                  <ArrowRight className="size-4" />
                </Link>
              }
            />
            <Button
              size="lg"
              variant="outline"
              className="rounded-xl"
              nativeButton={false}
              render={<Link href="/#how-it-works">How it works</Link>}
            />
          </div>
          <p className="text-sm text-muted-foreground">
            No account required · Takes about 2–4 minutes
          </p>
        </div>

        <div className="flex justify-center lg:justify-end">
          <ModalityVisual />
        </div>
      </div>
    </section>
  )
}