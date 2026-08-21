/**
 * Generic async hook for API calls with loading / error / result state.
 */

import { useState, useCallback } from "react";

interface UseAnalysisState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  run: (...args: Parameters<() => Promise<T>>) => void;
}

export function useAnalysis<T>(
  apiFn: (...args: unknown[]) => Promise<T>
): {
  data: T | null;
  loading: boolean;
  error: string | null;
  run: (...args: unknown[]) => void;
} {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(
    (...args: unknown[]) => {
      setLoading(true);
      setError(null);
      apiFn(...args)
        .then((result) => {
          setData(result);
          setLoading(false);
        })
        .catch((err: unknown) => {
          setError(err instanceof Error ? err.message : "Unknown error");
          setLoading(false);
        });
    },
    [apiFn]
  );

  return { data, loading, error, run };
}
