import { mapRecommendations } from '@/lib/api-mappers'
import type { Recommendation } from '@/lib/types'

export function SuggestionsSection({ recommendations }: { recommendations?: unknown }) {
  const items: Recommendation[] = mapRecommendations(recommendations)

  return (
    <div className="flex flex-col gap-4">
      {items.map((group) => (
        <div
          key={group.category}
          className="rounded-2xl border border-border bg-card p-5"
        >
          <h3 className="font-medium">{group.category}</h3>
          <ul className="mt-3 flex flex-col gap-2">
            {group.items.map((item) => (
              <li key={item} className="flex gap-2 text-sm text-muted-foreground">
                <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-primary" aria-hidden />
                <span className="text-pretty">{item}</span>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  )
}
