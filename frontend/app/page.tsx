import Link from 'next/link'
import { ArrowRight } from 'lucide-react'
import { SiteHeader } from '@/components/layout/site-header'
import { SiteFooter } from '@/components/layout/site-footer'
import { Hero } from '@/components/landing/hero'
import { FeatureCards } from '@/components/landing/feature-cards'
import { HowItWorks } from '@/components/landing/how-it-works'
import { ResearchSection } from '@/components/landing/research-section'
import { Disclaimer } from '@/components/shared/disclaimer'
import { Button } from '@/components/ui/button'

export default function HomePage() {
  return (
    <div className="flex min-h-dvh flex-col">
      <SiteHeader />
      <main className="flex-1">
        <Hero />
        <FeatureCards />
        <HowItWorks />
        <ResearchSection />

        <section className="mx-auto w-full max-w-[1216px] px-4 pb-16 sm:px-6">
          <div className="relative overflow-hidden rounded-3xl border border-border bg-card px-6 py-12 text-center sm:px-12 sm:py-16">
            <div className="pointer-events-none absolute inset-0 bg-grid opacity-30 [mask-image:radial-gradient(ellipse_at_center,black,transparent_70%)]" />
            <div className="relative mx-auto flex max-w-xl flex-col items-center gap-5">
              <h2 className="text-3xl font-semibold tracking-tight text-balance sm:text-4xl">
                Ready to begin your screening?
              </h2>
              <p className="text-muted-foreground text-pretty">
                Complete four short steps and receive an interpretable, multimodal screening
                result you can explore in detail.
              </p>
              <Link
                href="/screening"
                className="inline-flex h-9 items-center gap-1.5 rounded-xl bg-primary px-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/80"
              >
                Start screening
                <ArrowRight className="size-4" />
              </Link>
            </div>
          </div>
          <Disclaimer className="mt-8" />
        </section>
      </main>
      <SiteFooter />
    </div>
  )
}
