'use client'

import { FileText, Camera, ScanFace, AudioLines } from 'lucide-react'
import { LogoMark } from '@/components/layout/logo'

const NODES = [
  { key: 'text', label: 'Text', icon: FileText, x: 18, y: 20 },
  { key: 'photo', label: 'Photo', icon: Camera, x: 82, y: 20 },
  { key: 'face', label: 'Face', icon: ScanFace, x: 18, y: 80 },
  { key: 'voice', label: 'Voice', icon: AudioLines, x: 82, y: 80 },
] as const

export function ModalityVisual() {
  return (
    <div
      className="relative aspect-square w-full max-w-md select-none"
      aria-label="Diagram: text, photo, face, and voice signals converging into NeuroScreen"
      role="img"
    >
      {/* connecting lines */}
      <svg
        viewBox="0 0 100 100"
        className="absolute inset-0 size-full"
        fill="none"
        aria-hidden
      >
        <defs>
          <linearGradient id="lineGrad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="var(--primary)" stopOpacity="0.05" />
            <stop offset="50%" stopColor="var(--primary)" stopOpacity="0.45" />
            <stop offset="100%" stopColor="var(--ai)" stopOpacity="0.15" />
          </linearGradient>
        </defs>
        {NODES.map((n) => (
          <line
            key={n.key}
            x1={n.x}
            y1={n.y}
            x2={50}
            y2={50}
            stroke="url(#lineGrad)"
            strokeWidth="0.6"
            strokeLinecap="round"
          >
            <animate
              attributeName="stroke-opacity"
              values="0.4;1;0.4"
              dur="3.5s"
              repeatCount="indefinite"
            />
          </line>
        ))}
        {/* traveling pulses along each line */}
        {NODES.map((n, i) => (
          <circle key={`p-${n.key}`} r="0.9" fill="var(--ai)" className="motion-reduce:hidden">
            <animate
              attributeName="cx"
              values={`${n.x};50`}
              dur="2.8s"
              begin={`${i * 0.5}s`}
              repeatCount="indefinite"
            />
            <animate
              attributeName="cy"
              values={`${n.y};50`}
              dur="2.8s"
              begin={`${i * 0.5}s`}
              repeatCount="indefinite"
            />
            <animate
              attributeName="opacity"
              values="0;1;0"
              dur="2.8s"
              begin={`${i * 0.5}s`}
              repeatCount="indefinite"
            />
          </circle>
        ))}
      </svg>

      {/* center hub */}
      <div className="absolute left-1/2 top-1/2 flex -translate-x-1/2 -translate-y-1/2 flex-col items-center">
        <div className="relative flex size-24 items-center justify-center rounded-3xl border border-primary/20 bg-card shadow-[0_8px_40px_-12px] shadow-primary/25">
          <span className="absolute inset-0 animate-ping rounded-3xl bg-primary/5 [animation-duration:3s] motion-reduce:hidden" />
          <LogoMark className="size-11 rounded-2xl" />
        </div>
        <span className="mt-2 text-xs font-medium tracking-wide text-muted-foreground">
          NeuroScreen
        </span>
      </div>

      {/* modality nodes */}
      {NODES.map((n) => (
        <div
          key={n.key}
          className="absolute flex -translate-x-1/2 -translate-y-1/2 flex-col items-center gap-1.5"
          style={{ left: `${n.x}%`, top: `${n.y}%` }}
        >
          <div className="flex size-12 items-center justify-center rounded-2xl border border-border bg-card text-foreground/80 shadow-sm">
            <n.icon className="size-5" />
          </div>
          <span className="text-xs font-medium text-muted-foreground">{n.label}</span>
        </div>
      ))}
    </div>
  )
}
