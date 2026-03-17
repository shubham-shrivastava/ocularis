"use client";

import { useEffect, useRef, useState } from "react";
import type { StepTraceWire } from "@/lib/types";
import { wsUrl, screenshotUrl } from "@/lib/api";

interface Props {
  runId: string;
  initialSteps?: StepTraceWire[];
}

export function LiveView({ runId, initialSteps = [] }: Props) {
  const [steps, setSteps] = useState<StepTraceWire[]>(initialSteps);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const ws = new WebSocket(wsUrl(runId));
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === "ping") return;
        if (data.error) return;
        setSteps((prev) => {
          const next = data as StepTraceWire;
          const existing = prev.findIndex((step) => step.step_number === next.step_number);
          if (existing === -1) return [...prev, next];
          const copy = [...prev];
          copy[existing] = next;
          return copy;
        });
      } catch {
        // ignore malformed frames
      }
    };

    return () => ws.close();
  }, [runId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [steps]);

  const latest = steps[steps.length - 1];

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 text-xs text-zinc-500">
        <span
          className={`inline-block w-2 h-2 rounded-full ${connected ? "bg-green-500 animate-pulse" : "bg-zinc-600"}`}
        />
        {connected ? "Live" : "Disconnected"}
        <span className="ml-2">{steps.length} steps received</span>
      </div>

      {/* Screenshot viewer */}
      {latest?.post_state.screenshot_url ? (
        <div className="rounded-lg overflow-hidden border border-zinc-800">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={screenshotUrl(runId, latest.step_number, "post")}
            alt={`Step ${latest.step_number} post-action`}
            className="w-full"
          />
          <div className="bg-zinc-900 px-3 py-1.5 text-xs text-zinc-400 flex justify-between">
            <span>Step {latest.step_number} — {latest.post_state.url}</span>
            <span className={latest.outcome === "success" ? "text-green-400" : "text-yellow-400"}>
              {latest.outcome}
            </span>
          </div>
        </div>
      ) : (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 h-48 flex items-center justify-center text-zinc-600 text-sm">
          {steps.length === 0 ? "Waiting for first step..." : "No screenshot (ephemeral run)"}
        </div>
      )}

      {/* Step log */}
      <div className="rounded-lg border border-zinc-800 bg-zinc-900 max-h-48 overflow-y-auto text-xs font-mono">
        {steps.map((step) => (
          <div
            key={step.step_number}
            className="flex gap-3 px-3 py-1.5 border-b border-zinc-800 last:border-0 hover:bg-zinc-800/50"
          >
            <span className="text-zinc-600 w-6 text-right shrink-0">{step.step_number}</span>
            <span className="text-zinc-400">{step.action.action_type}</span>
            <span className="text-zinc-600 truncate">{JSON.stringify(step.action.params)}</span>
            <span
              className={`ml-auto shrink-0 ${
                step.outcome === "success"
                  ? "text-green-500"
                  : step.outcome === "error"
                  ? "text-red-500"
                  : "text-yellow-500"
              }`}
            >
              {step.outcome}
            </span>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
