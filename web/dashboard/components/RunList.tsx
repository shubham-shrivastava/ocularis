"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import type { RunAccepted } from "@/lib/types";
import { listRuns } from "@/lib/api";
import { StatusBadge } from "./StatusBadge";

export function RunList() {
  const [runs, setRuns] = useState<RunAccepted[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const data = await listRuns();
        setRuns(data);
      } catch {
        setRuns([]);
      } finally {
        setLoading(false);
      }
    };
    load();
    const interval = setInterval(load, 3000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return <p className="text-zinc-500 text-sm">Loading runs...</p>;
  }

  if (runs.length === 0) {
    return (
      <div className="rounded-lg border border-zinc-800 p-6 text-center text-zinc-500 text-sm">
        No runs yet. Start one via the API.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {runs.map((run) => (
        <Link
          key={run.run_id}
          href={`/runs/${run.run_id}`}
          className="flex items-center justify-between rounded-lg border border-zinc-800 bg-zinc-900 px-4 py-3 hover:border-zinc-600 transition-colors"
        >
          <span className="text-sm text-zinc-300 font-medium truncate max-w-[60%]">
            {run.run_id}
          </span>
          <StatusBadge status={run.status} />
        </Link>
      ))}
    </div>
  );
}
