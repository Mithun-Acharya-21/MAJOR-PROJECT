'use client'

import Link from 'next/link'
import { HelpCircle, User } from 'lucide-react'
import { Logo } from './logo'
import { ThemeToggle } from './theme-toggle'
import { Button } from '@/components/ui/button'

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 border-b border-border/70 bg-background/80 backdrop-blur-xl">
      <div className="mx-auto flex h-16 w-full max-w-[1216px] items-center justify-between px-4 sm:px-6">
        <Logo />
        <nav className="hidden items-center gap-1 md:flex">
          <Button
            variant="ghost"
            size="sm"
            nativeButton={false}
            render={<Link href="/#how-it-works">How it works</Link>}
          />
          <Button
            variant="ghost"
            size="sm"
            nativeButton={false}
            render={<Link href="/#research">Research</Link>}
          />
          <Button
            variant="ghost"
            size="sm"
            nativeButton={false}
            render={<Link href="/daic-woz">DAIC-WOZ Benchmark</Link>}
          />
          <Button
            variant="ghost"
            size="sm"
            nativeButton={false}
            render={<Link href="/doctor">Doctor Portal</Link>}
          />
        </nav>
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="icon" aria-label="Help">
            <HelpCircle className="size-4" />
          </Button>
          <ThemeToggle />
          <Button variant="ghost" size="icon" aria-label="Profile">
            <User className="size-4" />
          </Button>
          <Button
            size="sm"
            className="ml-1.5 rounded-xl"
            nativeButton={false}
            render={<Link href="/screening">Start screening</Link>}
          />
        </div>
      </div>
    </header>
  )
}