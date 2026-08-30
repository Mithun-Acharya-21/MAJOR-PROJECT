'use client'

import { ArrowLeft, ArrowRight } from 'lucide-react'
import { Button } from '@/components/ui/button'

export function StepFooter({
  onBack,
  onNext,
  nextDisabled,
  nextLabel = 'Continue',
  showBack = true,
  showNext = true,
}: {
  onBack?: () => void
  onNext?: () => void
  nextDisabled?: boolean
  nextLabel?: string
  showBack?: boolean
  showNext?: boolean
}) {
  return (
    <div className="flex items-center justify-between gap-3 pt-2">
      {showBack ? (
        <Button variant="ghost" onClick={onBack} className="gap-1.5">
          <ArrowLeft className="size-4" />
          Back
        </Button>
      ) : (
        <span />
      )}
      {showNext ? (
        <Button onClick={onNext} disabled={nextDisabled} className="rounded-xl">
          {nextLabel}
          <ArrowRight className="size-4" />
        </Button>
      ) : (
        <span />
      )}
    </div>
  )
}
