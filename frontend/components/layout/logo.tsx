import { cn } from '@/lib/utils'
import Link from 'next/link'

export function Logo({ className, href = '/' }: { className?: string; href?: string }) {
  return (
    <Link
      href={href}
      className={cn(
        'inline-flex items-center gap-2.5 rounded-lg outline-none focus-visible:ring-3 focus-visible:ring-ring/50',
        className,
      )}
    >
      <LogoMark />
      <span className="text-[0.95rem] font-semibold tracking-tight text-foreground">
        Depression<span className="text-primary">Detection</span>
      </span>
    </Link>
  )
}

export function LogoMark({ className }: { className?: string }) {
  return (
    <span
      className={cn(
        'relative flex size-8 items-center justify-center rounded-lg bg-primary/10 ring-1 ring-inset ring-primary/20',
        className,
      )}
      aria-hidden
    >
      <svg viewBox="0 0 24 24" fill="none" className="size-5">
        <path
          d="M12 3v4M12 17v4M3 12h4M17 12h4M6.3 6.3l2.8 2.8M14.9 14.9l2.8 2.8M17.7 6.3l-2.8 2.8M9.1 14.9l-2.8 2.8"
          stroke="var(--primary)"
          strokeWidth="1.5"
          strokeLinecap="round"
          opacity="0.55"
        />
        <circle cx="12" cy="12" r="3.2" fill="var(--primary)" />
        <circle cx="12" cy="3" r="1.4" fill="var(--ai)" />
        <circle cx="21" cy="12" r="1.4" fill="var(--ai)" />
        <circle cx="12" cy="21" r="1.4" fill="var(--primary)" />
        <circle cx="3" cy="12" r="1.4" fill="var(--primary)" />
      </svg>
    </span>
  )
}