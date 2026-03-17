"use client";

import { useEffect, useState } from "react";
import { getRun, pauseRun, resumeRun, interveneRun } from "@/lib/api";
import type { RunStatus } from "@/lib/types";

interface Props {
  runId: string;
  status: RunStatus;
}

export function HitlControls({ runId, status }: Props) {
  const [runStatus, setRunStatus] = useState<RunStatus>(status);
  const [waitingReason, setWaitingReason] = useState<string | null>(null);
  const [clickX, setClickX] = useState("");
  const [clickY, setClickY] = useState("");
  const [typeText, setTypeText] = useState("");
  const [loginUserId, setLoginUserId] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const run = await getRun(runId);
        if (!cancelled) {
          setRunStatus(run.status);
          setWaitingReason(run.waiting_reason ?? null);
        }
      } catch {
        // keep last known status
      }
    };

    poll();
    const timer = setInterval(poll, 2000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [runId]);

  const flash = (msg: string) => {
    setMessage(msg);
    setTimeout(() => setMessage(""), 3000);
  };

  const handle = async (fn: () => Promise<void>, label: string) => {
    setBusy(true);
    try {
      await fn();
      flash(`${label} sent`);
    } catch {
      flash(`${label} failed`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-4 rounded-lg border border-zinc-800 bg-zinc-900 p-4">
      <h3 className="text-sm font-semibold text-zinc-300">Human Controls</h3>

      {runStatus === "waiting_for_human" && waitingReason === "login_required" && (
        <div className="rounded border border-yellow-800 bg-yellow-950/60 p-3 space-y-2">
          <p className="text-xs text-yellow-300 font-medium">
            Agent is waiting for login credentials
          </p>
          <p className="text-xs text-zinc-400">
            Fill credentials below. Password is masked in UI and redacted from logs/traces.
          </p>
          <div className="space-y-2">
            <input
              type="text"
              placeholder="User ID / Email"
              value={loginUserId}
              onChange={(e) => setLoginUserId(e.target.value)}
              className="flex-1 rounded bg-zinc-800 border border-zinc-700 px-2 py-1 text-xs text-zinc-200 focus:outline-none focus:border-zinc-500"
            />
            <input
              type="password"
              placeholder="Password"
              value={loginPassword}
              onChange={(e) => setLoginPassword(e.target.value)}
              className="flex-1 rounded bg-zinc-800 border border-zinc-700 px-2 py-1 text-xs text-zinc-200 focus:outline-none focus:border-zinc-500"
            />
          </div>
          <div className="flex gap-2 items-center">
            <button
              disabled={busy || !loginUserId.trim() || !loginPassword}
              onClick={() =>
                handle(
                  () =>
                    interveneRun(runId, {
                      action_type: "login",
                      params: {
                        user_id: loginUserId.trim(),
                        password: loginPassword,
                      },
                    }),
                  "Credentials"
                )
              }
              className="px-3 py-1 text-xs rounded bg-indigo-700 hover:bg-indigo-600 disabled:opacity-40 disabled:cursor-not-allowed text-white transition-colors"
            >
              Submit Credentials
            </button>
          </div>
        </div>
      )}

      {/* Pause / Resume */}
      <div className="flex gap-2">
        <button
          disabled={busy || runStatus !== "running"}
          onClick={() => handle(() => pauseRun(runId), "Pause")}
          className="px-3 py-1.5 text-xs rounded bg-yellow-700 hover:bg-yellow-600 disabled:opacity-40 disabled:cursor-not-allowed text-white transition-colors"
        >
          Pause
        </button>
        <button
          disabled={busy || runStatus !== "paused"}
          onClick={() => handle(() => resumeRun(runId), "Resume")}
          className="px-3 py-1.5 text-xs rounded bg-green-700 hover:bg-green-600 disabled:opacity-40 disabled:cursor-not-allowed text-white transition-colors"
        >
          Resume
        </button>
      </div>

      {/* Manual click intervention */}
      {!(runStatus === "waiting_for_human" && waitingReason === "login_required") && (
        <>
          <div className="space-y-1">
            <label className="text-xs text-zinc-500">Click at coordinates</label>
            <div className="flex gap-2 items-center">
              <input
                type="number"
                placeholder="X"
                value={clickX}
                onChange={(e) => setClickX(e.target.value)}
                className="w-20 rounded bg-zinc-800 border border-zinc-700 px-2 py-1 text-xs text-zinc-200 focus:outline-none focus:border-zinc-500"
              />
              <input
                type="number"
                placeholder="Y"
                value={clickY}
                onChange={(e) => setClickY(e.target.value)}
                className="w-20 rounded bg-zinc-800 border border-zinc-700 px-2 py-1 text-xs text-zinc-200 focus:outline-none focus:border-zinc-500"
              />
              <button
                disabled={busy || !clickX || !clickY}
                onClick={() =>
                  handle(
                    () =>
                      interveneRun(runId, {
                        action_type: "click",
                        params: { x: Number(clickX), y: Number(clickY) },
                      }),
                    "Click"
                  )
                }
                className="px-3 py-1 text-xs rounded bg-blue-700 hover:bg-blue-600 disabled:opacity-40 disabled:cursor-not-allowed text-white transition-colors"
              >
                Send Click
              </button>
            </div>
          </div>

          {/* Manual type intervention */}
          <div className="space-y-1">
            <label className="text-xs text-zinc-500">Type text</label>
            <div className="flex gap-2 items-center">
              <input
                type="text"
                placeholder="text to type..."
                value={typeText}
                onChange={(e) => setTypeText(e.target.value)}
                className="flex-1 rounded bg-zinc-800 border border-zinc-700 px-2 py-1 text-xs text-zinc-200 focus:outline-none focus:border-zinc-500"
              />
              <button
                disabled={busy || !typeText}
                onClick={() =>
                  handle(
                    () =>
                      interveneRun(runId, {
                        action_type: "type",
                        params: { text: typeText },
                      }),
                    "Type"
                  )
                }
                className="px-3 py-1 text-xs rounded bg-blue-700 hover:bg-blue-600 disabled:opacity-40 disabled:cursor-not-allowed text-white transition-colors"
              >
                Send Text
              </button>
            </div>
          </div>
        </>
      )}

      {message && (
        <p className="text-xs text-green-400">{message}</p>
      )}
    </div>
  );
}
