import { SiteHeader } from '@/components/layout/site-header'
import { SiteFooter } from '@/components/layout/site-footer'
import { DaicWozResearchScreen } from '@/components/daic-woz/daic-woz-screen'

export const metadata = {
  title: 'DAIC-WOZ Multimodal Research Pipeline | NeuroScreen',
  description: 'Participant-level multimodal depression screening benchmark using DAIC-WOZ dataset.',
}

export default function DaicWozPage() {
  return (
    <div className="flex min-h-dvh flex-col bg-background">
      <SiteHeader />
      <main className="flex-1 py-6">
        <DaicWozResearchScreen />
      </main>
      <SiteFooter />
    </div>
  )
}
