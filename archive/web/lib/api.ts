export function apiUrl(): string {
  const fromServer = process.env.API_URL;
  const fromPublic = process.env.NEXT_PUBLIC_API_URL;
  const raw =
    (typeof window === "undefined" && fromServer) || fromPublic || "http://localhost:8000";
  return raw.replace(/\/$/, "");
}

export type NumberStat = { label: string; value: string | number | null };
export type Report = {
  numbers: NumberStat[];
  findings: string[];
  notes: string[];
};

export type StageBlock = {
  id: string;
  title: string;
  summary: string;
  checkpoint: boolean;
  status: string;
  report: Report | null;
};

export type Board = {
  id: string;
  status: string;
  mode: string;
  current_stage: string;
  checkpoints: string[];
  objective: {
    text?: string;
    target_column?: string;
    metric?: string;
    target?: number;
  };
  error: string | null;
  pending_approval: { tool?: string } | null;
  target_met: boolean;
  best: { model?: string; metrics?: { f1?: number } } | null;
  notes: string[];
  sme_notes: { stage: string; prompt: string }[];
  stages: StageBlock[];
  can_next: boolean;
  can_retry?: boolean;
  can_rework: boolean;
  can_approve: boolean;
  can_choose_model: boolean;
  choices: ModelChoice[];
  chosen_models: string[];
};

export type ModelChoice = {
  id: string;
  title: string;
  reason: string;
  blurb?: string;
};

export type ExperimentSummary = {
  id: string;
  status: string;
  mode?: string;
  current_stage?: string;
  objective?: string;
  target_met?: boolean;
  model?: string;
  f1?: number;
};

async function parse<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = (body as { error?: string }).error || response.statusText;
    throw new Error(message);
  }
  return body as T;
}

export async function listExperiments(): Promise<ExperimentSummary[]> {
  const response = await fetch(`${apiUrl()}/experiments`, { cache: "no-store" });
  return parse(response);
}

export async function clearExperiments(): Promise<{ removed: number }> {
  const response = await fetch(`${apiUrl()}/experiments`, { method: "DELETE" });
  return parse(response);
}

export function asBoard(value: Board): Board {
  const rawStages = (value as { stages?: unknown }).stages;
  const stages = Array.isArray(rawStages) ? rawStages : [];
  return {
    ...value,
    status: value.status || "unknown",
    stages: stages.map((stage) => ({
      ...stage,
      report: stage.report
        ? {
            numbers: Array.isArray(stage.report.numbers) ? stage.report.numbers : [],
            findings: Array.isArray(stage.report.findings) ? stage.report.findings : [],
            notes: Array.isArray(stage.report.notes) ? stage.report.notes : [],
          }
        : null,
    })),
    choices: Array.isArray(value.choices) ? value.choices : [],
    chosen_models: Array.isArray(value.chosen_models) ? value.chosen_models : [],
    notes: Array.isArray(value.notes) ? value.notes : [],
    sme_notes: Array.isArray(value.sme_notes) ? value.sme_notes : [],
  };
}

export async function getBoard(id: string): Promise<Board> {
  const response = await fetch(`${apiUrl()}/experiments/${id}/board`, {
    cache: "no-store",
  });
  return asBoard(await parse<Board>(response));
}

export async function startRun(input: {
  file: File;
  objective: string;
}): Promise<Board> {
  const form = new FormData();
  form.set("objective", input.objective);
  form.set("target_column", "");
  form.set("metric", "");
  form.set("metrics", "");
  form.set("approve_training", "true");
  form.set("staged", "true");
  form.set("file", input.file);
  const response = await fetch(`${apiUrl()}/experiments/upload`, {
    method: "POST",
    body: form,
  });
  return parse(response);
}

export async function resumeStage(id: string): Promise<Board> {
  const response = await fetch(`${apiUrl()}/experiments/${id}/resume`, {
    method: "POST",
  });
  return parse(response);
}

export async function nextStage(id: string): Promise<Board> {
  const response = await fetch(`${apiUrl()}/experiments/${id}/next`, {
    method: "POST",
  });
  return parse(response);
}

export async function reworkStage(id: string, prompt: string): Promise<Board> {
  const response = await fetch(`${apiUrl()}/experiments/${id}/rework`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  return parse(response);
}

export async function chooseModel(id: string, model: string): Promise<Board> {
  const response = await fetch(`${apiUrl()}/experiments/${id}/choose-model`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  });
  return parse(response);
}

export async function approveTraining(id: string, confirmed: boolean): Promise<Board> {
  const response = await fetch(`${apiUrl()}/experiments/${id}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirmed }),
  });
  return parse(response);
}
