import { LiveView } from "@/components/LiveView";
import { HitlControls } from "@/components/HitlControls";
import { StatusBadge } from "@/components/StatusBadge";
import { getRun } from "@/lib/api";
import type { RunDetail } from "@/lib/types";
import Link from "next/link";

interface Props {
  params: { runId: string };
}

export default async function RunPage({ params }: Props) {
  let run: RunDetail | null = null;
  let error = "";

  try {
    run = await getRun(params.runId);
  } catch {
    error = "Could not load run. The API may be offline.";
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Link href="/" className="text-zinc-500 hover:text-zinc-300 text-sm transition-colors">
          ← Runs
        </Link>
        {run && <StatusBadge status={run.status} />}
      </div>

      {error && (
        <div className="rounded-lg border border-red-900 bg-red-950 px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {run && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: Run info + HITL controls */}
          <div className="space-y-4">
            <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4 space-y-2">
              <h2 className="text-sm font-semibold text-zinc-300">Run Details</h2>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-zinc-500">ID</span>
                  <span className="text-zinc-300 font-mono truncate max-w-[60%]">{run.run_id}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Mode</span>
                  <span className="text-zinc-300">{run.comparison_mode}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Ephemeral</span>
                  <span className="text-zinc-300">{run.ephemeral ? "yes" : "no"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Steps</span>
                  <span className="text-zinc-300">{run.steps.length}</span>
                </div>
              </div>
            </div>

            <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
              <h2 className="text-sm font-semibold text-zinc-300 mb-1">Goal</h2>
              <p className="text-xs text-zinc-400 leading-relaxed">{run.goal}</p>
            </div>

            {run.comparison_state && (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4 space-y-2">
                <div className="flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-zinc-300">Comparison Progress</h2>
                  <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 capitalize">
                    {run.comparison_state.status}
                  </span>
                </div>
                <div className="w-full bg-zinc-800 rounded-full h-1.5">
                  <div
                    className="bg-blue-500 h-1.5 rounded-full transition-all"
                    style={{ width: `${Math.min(100, (run.comparison_state.collected_count / Math.max(1, run.comparison_state.target_count)) * 100)}%` }}
                  />
                </div>
                <p className="text-xs text-zinc-500">
                  {run.comparison_state.collected_count} / {run.comparison_state.target_count} candidates collected
                </p>
                {run.comparison_state.collected_items.length > 0 && (
                  <div className="space-y-1 pt-1">
                    {run.comparison_state.collected_items.slice(0, 6).map((item, i) => (
                      <a
                        key={`${item.url}-${item.title}-${i}`}
                        href={item.url}
                        target="_blank"
                        rel="noreferrer"
                        className="flex items-center justify-between text-xs text-zinc-400 hover:text-zinc-200 transition-colors"
                      >
                        <span className="truncate mr-2">{item.title}</span>
                        {item.price && (
                          <span className="shrink-0 text-emerald-400/70 text-[10px]">{item.price}</span>
                        )}
                      </a>
                    ))}
                  </div>
                )}
              </div>
            )}

            {run.result?.answer && (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-zinc-300">Final Answer</h2>
                  <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400">
                    {Math.round(run.result.answer.confidence * 100)}% confidence
                  </span>
                </div>

                {run.result.answer.text && (
                  <div className="rounded-md bg-emerald-950/50 border border-emerald-900/50 px-3 py-2">
                    <p className="text-sm text-emerald-300 font-medium leading-relaxed">
                      {run.result.answer.text}
                    </p>
                  </div>
                )}

                {run.result.answer.link && (
                  <a
                    href={run.result.answer.link}
                    className="block text-xs text-blue-400 hover:text-blue-300 underline break-all transition-colors"
                    target="_blank"
                    rel="noreferrer"
                  >
                    {run.result.answer.link}
                  </a>
                )}

                {run.result.answer.items.length > 0 && (
                  <div className="space-y-2 pt-1">
                    <h3 className="text-xs font-medium text-zinc-500 uppercase tracking-wide">
                      Compared {run.result.answer.items.length} products
                    </h3>
                    {run.result.answer.items.map((item, i) => (
                      <a
                        key={`${item.url}-${item.title}-${i}`}
                        href={item.url}
                        target="_blank"
                        rel="noreferrer"
                        className="block rounded-md bg-zinc-800/70 hover:bg-zinc-800 border border-zinc-700/50 hover:border-zinc-600 px-3 py-2.5 transition-colors group"
                      >
                        <div className="text-xs text-zinc-200 group-hover:text-white leading-snug">
                          {item.title}
                        </div>
                        <div className="flex items-center gap-3 mt-1.5">
                          {item.price && (
                            <span className="text-xs font-semibold text-emerald-400">{item.price}</span>
                          )}
                          {item.rating && (
                            <span className="text-[10px] text-amber-400/80 flex items-center gap-0.5">
                              <svg className="w-3 h-3 fill-current" viewBox="0 0 20 20"><path d="M10 15l-5.878 3.09 1.123-6.545L.49 6.91l6.564-.955L10 0l2.946 5.955 6.564.955-4.755 4.635 1.123 6.545z"/></svg>
                              {item.rating}
                            </span>
                          )}
                          <span className="text-[10px] text-blue-400/60 ml-auto group-hover:text-blue-400 transition-colors">
                            View →
                          </span>
                        </div>
                      </a>
                    ))}
                  </div>
                )}
              </div>
            )}

            {run.goal_verdict && (
              <div className={`rounded-lg border p-4 ${
                run.goal_verdict.status === "achieved"
                  ? "border-green-800 bg-green-950"
                  : run.goal_verdict.status === "not_achieved"
                  ? "border-red-900 bg-red-950"
                  : "border-yellow-800 bg-yellow-950"
              }`}>
                <h2 className="text-sm font-semibold text-zinc-300 mb-1">Goal Verdict</h2>
                <p className="text-xs font-medium capitalize mb-1">{run.goal_verdict.status.replace(/_/g, " ")}</p>
                <p className="text-xs text-zinc-400 mb-1">
                  Confidence: {Math.round(run.goal_verdict.confidence * 100)}%
                </p>
                <p className="text-xs text-zinc-500">{run.goal_verdict.reasoning}</p>
              </div>
            )}

            <HitlControls runId={run.run_id} status={run.status} />
          </div>

          {/* Right: Live view (2/3 width) */}
          <div className="lg:col-span-2">
            <h2 className="text-sm font-semibold text-zinc-300 mb-3">Live View</h2>
            <LiveView runId={run.run_id} initialSteps={run.steps} />
          </div>
        </div>
      )}
    </div>
  );
}
