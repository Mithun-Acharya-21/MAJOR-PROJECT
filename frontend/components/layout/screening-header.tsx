'use client'

import Link from 'next/link'
import { X } from 'lucide-react'
import { Logo } from './logo'
import { ThemeToggle } from './theme-toggle'
import { Button } from '@/components/ui/button'

export function ScreeningHeader({
  currentStep,
  totalSteps,
}: {
  currentStep: number
  totalSteps: number
}) {
  return (
    <header className="sticky top-0 z-40 border-b border-border/70 bg-background/80 backdrop-blur-xl">
      <div className="mx-auto flex h-16 w-full max-w-[1216px] items-center justify-between px-4 sm:px-6">
        <Logo />
        <div className="flex items-center gap-3 sm:gap-4">
          <span
            className="text-sm font-medium text-muted-foreground tabular-nums"
            aria-live="polite"
          >
            Step {currentStep} of {totalSteps}
          </span>
          <ThemeToggle />
          <Button variant="ghost" size="sm" className="gap-1.5" asChild>
            <Link href="/">
              <X className="size-4" />
              <span className="hidden sm:inline">Exit</span>
            </Link>
          </Button>
        </div>
      </div>
    </header>
  )
}
