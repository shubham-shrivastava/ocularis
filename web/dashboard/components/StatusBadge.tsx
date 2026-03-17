import type { RunStatus } from "@/lib/types";

const COLOR: Record<RunStatus, string> = {
  queued: "bg-zinc-700 text-zinc-300",
  running: "bg-blue-600 text-white animate-pulse",
  paused: "bg-yellow-600 text-white",
  waiting_for_human: "bg-orange-600 text-white animate-pulse",
  completed: "bg-green-700 text-white",
  max_steps_exceeded: "bg-purple-700 text-white",
  failed: "bg-red-700 text-white",
};

export function StatusBadge({ status }: { status: RunStatus }) {
  return (
    <span className={`inline-flex items-center rounded px-2 py-0.5 text-xs font-medium ${COLOR[status]}`}>
      {status.replace(/_/g, " ")}
    </span>
  );
}
