import { LogoMark } from './logo'

export function SiteFooter() {
  return (
    <footer className="border-t border-border/70">
      <div className="mx-auto w-full max-w-[1216px] px-4 py-10 sm:px-6">
        <div className="flex flex-col gap-6 sm:flex-row sm:items-start sm:justify-between">
          <div className="flex items-center gap-2.5">
            <LogoMark />
            <span className="text-sm font-semibold tracking-tight">
              Neuro<span className="text-primary">Screen</span>
            </span>
          </div>
          <p className="max-w-md text-sm leading-relaxed text-muted-foreground text-pretty">
            NeuroScreen is a research and educational screening tool. It does not provide a
            medical diagnosis, and its results are model-generated indicators rather than
            clinical conclusions.
          </p>
        </div>
        <p className="mt-8 text-xs text-muted-foreground/70">
          For research and educational use. Built as a multimodal AI demonstration.
        </p>
      </div>
    </footer>
  )
}
