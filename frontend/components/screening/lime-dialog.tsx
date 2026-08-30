'use client'

import { Modal } from '@/components/ui/modal'
import { LimeView } from './lime-view'
import type { LimeExplanation } from '@/lib/types'

export function LimeDialog({
  open,
  onOpenChange,
  explanation,
  title,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  explanation: LimeExplanation
  title: string
}) {
  return (
    <Modal
      open={open}
      onOpenChange={onOpenChange}
      title={title}
      description="Model features that contributed most to this screening result."
    >
      <LimeView explanation={explanation} />
    </Modal>
  )
}
