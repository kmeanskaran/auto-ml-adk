// The shape of GET /api/view (app/ui/routes.py).

export type Scores = Record<string, number | null>;

export type ChartSpec = {
  type: "bar" | "hbar" | "line" | "stacked";
  title: string;
  categories: string[];
  series: { name: string; values: (number | null)[] }[];
  domain?: [number, number];
  x_label?: string;
  y_label?: string;
};

export type MetricInfo = {
  label: string;
  plain: string;
  about: string;
  kind: "rate" | "score" | "cost";
  higher_is_better: boolean;
};

export type Review = {
  verdict?: "pass" | "concerns";
  findings?: string[];
  recommendations?: string[];
  recommended_model?: string;
};

export type Candidate = {
  model: string;
  label: string;
  threshold: number;
  valid: Scores;
  test: Scores;
};

export type Leaderboard = {
  metric: string;
  metric_label: string;
  best: string;
  baseline: { threshold: number; valid: Scores; test: Scores };
  warnings: string[];
  protocol: string;
  chosen_on: string;
  rows: Candidate[];
};

export type ColumnProfile = {
  name: string;
  kind: string;
  missing_pct: number;
  unique: number;
  signal_auc: number | null;
  stats?: { min: number; max: number };
  top?: [string, number][];
};

export type Profile = {
  rows: number;
  columns: number;
  duplicate_rows: number;
  target: { positive_rate: number };
  column_profiles: ColumnProfile[];
  charts?: ChartSpec[];
};

export type FeatureRow = {
  name: string;
  description: string;
  source?: string[];
  null_pct?: number;
  signal_auc?: number | null;
  known_at_prediction?: string;
};

export type FeatureDetails = {
  features: FeatureRow[];
  tables?: Record<string, { rows: number; outcome_rate: number }>;
  split?: { method: string; reason: string };
  rows_removed?: Record<string, number>;
  excluded_columns?: Record<string, string>;
  store?: { view: string; version: string };
};

export type VersionCard = {
  version: string;
  status: string;
  model: string;
  model_key: string;
  metric: string;
  metric_label: string;
  test: Scores;
  baseline_test: Scores;
  threshold: number;
  feature_view?: { view: string; version: string };
  run: string;
  created_at: string;
  promoted_at?: string;
};

export type StageState = "waiting" | "working" | "reviewing" | "your_review" | "done" | "kept" | "discarded";

export type Stage = {
  key: "profile" | "features" | "plan" | "model" | "promote";
  label: string;
  agent: string;
  state: StageState;
  headline?: string;
  findings: string[];
  charts?: ChartSpec[];
  details?: Profile | FeatureDetails | Leaderboard | null;
  review?: Review | null;
  problems?: string[];
  proposal?: { models?: string[]; metric?: string; reason?: string };
  options?: { models: { key: string; label: string }[] };
  leaderboard?: Leaderboard | null;
  live?: VersionCard | null;
  decided_by?: string;
  rounds?: number;
  reused_from?: string | null; // profile: copied from this run (no feedback, same data)
};

export type Config = {
  goal: string;
  record: string;
  positive: string;
  false_alarm: string;
  ask_human: string[];
  self_review_rounds: number;
  dataset: string;
  target: string;
  feature_view: string;
  costs: { missed: number; false_alarm: number };
};

// What a run cost, from its trace (app/harness/trace.py usage()).
export type UsageTurn = { agent: string; seconds: number; tool_calls: number; tokens: number; cached_tokens: number };
export type Usage = {
  minutes: number; // wall clock, waits for you included
  agent_minutes: number; // the agents' own working time
  tokens: number;
  cached_tokens: number;
  cached_pct: number;
  tool_calls: number;
  turns?: UsageTurn[];
};

export type Pipeline = {
  status: "idle" | "running" | "paused" | "waiting" | "done" | "error";
  error: string;
  run: string;
  waiting_on: string;
  why: string;
  activity: string[];
  brief?: { feedback: string; builds_on: string | null } | null;
  next_builds_on: string | null;
  usage?: Usage | null;
  config: Config;
  stages: Stage[];
};

export type ChatMessage = { role: "you" | "analyst" | "error"; text: string; charts?: ChartSpec[] };

export type StoredFeature = {
  name: string;
  description: string;
  source: string[];
  dtype?: string;
  null_pct?: number | null;
  signal_auc?: number | null;
  known_at_prediction?: string;
  protected: string[];
};

export type FeatureVersion = {
  version: string;
  created_at: string;
  run: string;
  features: number;
  split?: string | null;
  tables: Record<string, { rows: number; outcome_rate: number }>;
  excluded_columns: Record<string, string>;
  columns: StoredFeature[];
  used_by: { version: string; status: string }[];
  live: boolean;
};

export type FeatureView = { view: string; versions: FeatureVersion[] };

export type RunCard = {
  run: string;
  at: string;
  outcome: string;
  feature_view?: string | null;
  features?: number;
  models?: string[] | null;
  metric?: string | null;
  best?: string | null;
  test_score?: number | null;
  version?: string | null;
  human_overrides: number;
  decisions: number;
  warnings: number;
  feedback?: string;
  builds_on?: string | null;
  usage?: Usage | null; // none for runs recorded before usage was
};

export type ModelView = {
  model: string;
  thinking: string;
  available: boolean;
  base: string;
  analyst: { model: string; thinking: string };
  models: { key: string; label: string }[];
  thinking_levels: string[];
};

export type View = {
  pipeline: Pipeline;
  registry: { production: VersionCard | null; versions: VersionCard[] };
  feature_store: FeatureView[];
  history: RunCard[];
  metrics: Record<string, MetricInfo>;
  model: ModelView;
  chat: { status: string; messages: ChatMessage[] };
};

export type Answer = {
  choice: string;
  text?: string;
  apply?: number[];
  models?: string[];
  metric?: string;
  model?: string;
};
