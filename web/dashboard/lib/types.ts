export type RunStatus =
  | "queued"
  | "running"
  | "paused"
  | "waiting_for_human"
  | "completed"
  | "max_steps_exceeded"
  | "failed";

export interface RunAccepted {
  run_id: string;
  status: RunStatus;
}

export interface AgentStateWire {
  url: string;
  timestamp: string;
  state_hash: string;
  screenshot_url: string | null;
}

export interface ActionRequest {
  action_type: "click" | "type" | "login" | "scroll" | "wait" | "key_press";
  params: Record<string, unknown>;
}

export interface StepTraceWire {
  step_number: number;
  pre_state: AgentStateWire;
  post_state: AgentStateWire;
  action: ActionRequest;
  outcome: "success" | "no_change" | "error";
  recovery_used: string | null;
  duration_ms: number;
  critic_analysis: string | null;
}

export interface GoalVerdict {
  status: "achieved" | "not_achieved" | "uncertain";
  confidence: number;
  reasoning: string;
}

export interface ResultCandidate {
  title: string;
  url: string;
  price?: string | null;
  rating?: string | null;
  snippet?: string;
  fields?: Record<string, string>;
}

export interface RunAnswer {
  result_type: "link" | "text" | "list" | "count";
  link?: string | null;
  text?: string | null;
  items: ResultCandidate[];
  confidence: number;
}

export interface ComparisonState {
  target_count: number;
  collected_count: number;
  collected_items: ResultCandidate[];
  compared_items: ResultCandidate[];
  selected_item?: ResultCandidate | null;
  status: "collecting" | "ready" | "answered";
}

export interface RunResult {
  summary: string;
  final_url?: string | null;
  requested_count?: number | null;
  collected_count: number;
  candidates: ResultCandidate[];
  answer?: RunAnswer | null;
}

export interface RunDetail {
  run_id: string;
  goal: string;
  start_url: string;
  status: RunStatus;
  comparison_mode: string;
  ephemeral: boolean;
  steps: StepTraceWire[];
  sub_steps?: string[];
  current_sub_step?: number;
  waiting_reason?: string | null;
  goal_verdict: GoalVerdict | null;
  result?: RunResult | null;
  comparison_state?: ComparisonState | null;
  created_at: string | null;
  completed_at: string | null;
}
