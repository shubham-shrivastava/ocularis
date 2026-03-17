const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function listRuns(status?: string) {
  const url = status ? `${BASE}/runs?status=${status}` : `${BASE}/runs`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`listRuns failed: ${res.status}`);
  return res.json();
}

export async function getRun(runId: string) {
  const res = await fetch(`${BASE}/runs/${runId}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`getRun failed: ${res.status}`);
  return res.json();
}

export async function pauseRun(runId: string) {
  await fetch(`${BASE}/runs/${runId}/pause`, { method: "POST" });
}

export async function resumeRun(runId: string) {
  await fetch(`${BASE}/runs/${runId}/resume`, { method: "POST" });
}

export async function interveneRun(
  runId: string,
  action: { action_type: string; params: Record<string, unknown> }
) {
  await fetch(`${BASE}/runs/${runId}/intervene`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(action),
  });
}

export function screenshotUrl(runId: string, step: number, phase: "pre" | "post") {
  return `${BASE}/runs/${runId}/steps/${step}/screenshot?phase=${phase}`;
}

export function wsUrl(runId: string) {
  const base = BASE.replace(/^http/, "ws");
  return `${base}/ws/runs/${runId}`;
}
