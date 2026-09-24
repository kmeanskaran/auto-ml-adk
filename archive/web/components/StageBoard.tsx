"use client";

import { useEffect, useState } from "react";
import {
  approveTraining,
  asBoard,
  chooseModel,
  getBoard,
  nextStage,
  resumeStage,
  reworkStage,
  type Board,
  type ModelChoice,
  type Report,
  type StageBlock,
} from "@/lib/api";

const ACCENT_STATS = new Set(["Categorical", "Outliers", "F1", "Missing cells"]);

const RAIL_TITLE: Record<string, string> = {
  stats: "Statistics",
  prepare: "Prepare",
  model: "Features",
  train: "Training",
  deliver: "Delivery",
};

const STATE_LABEL: Record<string, string> = {
  pending: "Waiting",
  review: "In review",
  done: "Done",
  running: "Working",
  failed: "Failed",
  interrupted: "Stopped",
  awaiting_approval: "Approval",
  awaiting_review: "In review",
  completed: "Done",
};

export default function StageBoard({ initial }: { initial: Board }) {
  const [board, setBoard] = useState(() => asBoard(initial));
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (board.status !== "running") {
      return;
    }
    const timer = window.setInterval(async () => {
      try {
        setBoard(await getBoard(board.id));
      } catch {
        /* keep the last frame */
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [board.id, board.status]);

  async function run(action: () => Promise<Board>) {
    setBusy(true);
    setError("");
    try {
      const next = asBoard(await action());
      setBoard(next.stages.length === 0 && board.id ? await getBoard(board.id) : next);
      setPrompt("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed.");
    } finally {
      setBusy(false);
    }
  }

  const status = (board.status || "unknown").replaceAll("_", " ");
  return (
    <div className="desk">
      <header className="desk-head">
        <div>
          <p className="eyebrow">Run {board.id}</p>
          <h1>{board.objective?.text || "Run"}</h1>
        </div>
        <span className={`pill ${board.status}`}>{status}</span>
      </header>
      <ol className="rail">
        {board.stages.map((stage, index) => {
          const current = board.current_stage === stage.id;
          return (
            <li
              key={stage.id}
              className={["rail-step", current ? "current" : "", stage.status].join(" ")}
            >
              <span className="rail-index">{String(index + 1).padStart(2, "0")}</span>
              <span className="rail-title">{RAIL_TITLE[stage.id] || stage.title}</span>
              <span className="rail-state">{STATE_LABEL[stage.status] || stage.status}</span>
            </li>
          );
        })}
      </ol>
      {board.error ? <div className="error">{board.error}</div> : null}
      {board.stages.map((stage) =>
        stage.status === "pending" ? null : (
          <Stage
            key={stage.id}
            stage={stage}
            current={board.current_stage === stage.id}
            canNext={board.can_next && board.current_stage === stage.id}
            canRework={board.can_rework && board.current_stage === stage.id}
            canApprove={board.can_approve && board.current_stage === stage.id}
            canChoose={board.can_choose_model && board.current_stage === stage.id}
            canRetry={Boolean(board.can_retry) && board.current_stage === stage.id}
            choices={board.current_stage === stage.id ? board.choices : []}
            chosen={board.chosen_models}
            prompt={prompt}
            setPrompt={setPrompt}
            busy={busy}
            onNext={() => run(() => nextStage(board.id))}
            onRework={() => run(() => reworkStage(board.id, prompt))}
            onApprove={(confirmed) => run(() => approveTraining(board.id, confirmed))}
            onChoose={(model) => run(() => chooseModel(board.id, model))}
            onRetry={() => run(() => resumeStage(board.id))}
          />
        ),
      )}
      {error ? <div className="error">{error}</div> : null}
    </div>
  );
}

function ReportView({ stageId, report }: { stageId: string; report: Report }) {
  const numbers = report.numbers;
  const findings = report.findings;
  const notes = report.notes;
  const stats = (
    <div className="stats">
      {numbers.map((item) => (
        <div
          className={ACCENT_STATS.has(item.label) ? "stat emph" : "stat"}
          key={item.label}
        >
          <div className="k">{item.label}</div>
          <div className="v">{item.value ?? "—"}</div>
        </div>
      ))}
    </div>
  );
  const body = (
    <>
      {findings.length > 0 ? (
        <ul className="findings">
          {findings.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      ) : null}
      {notes.map((note) => (
        <div className="note" key={note}>
          {note}
        </div>
      ))}
    </>
  );
  if (stageId !== "stats") {
    return (
      <div className="brief">
        {stats}
        {body}
      </div>
    );
  }
  return (
    <div className="stats-section">
      <div className="stats-kicker">Statistical brief</div>
      <p className="stats-lead">Read the table before a model is allowed to touch it.</p>
      {stats}
      {body}
    </div>
  );
}

function Stage({
  stage,
  current,
  canNext,
  canRework,
  canApprove,
  canChoose,
  canRetry,
  choices,
  chosen,
  prompt,
  setPrompt,
  busy,
  onNext,
  onRework,
  onApprove,
  onChoose,
  onRetry,
}: {
  stage: StageBlock;
  current: boolean;
  canNext: boolean;
  canRework: boolean;
  canApprove: boolean;
  canChoose: boolean;
  canRetry: boolean;
  choices: ModelChoice[];
  chosen: string[];
  prompt: string;
  setPrompt: (value: string) => void;
  busy: boolean;
  onNext: () => void;
  onRework: () => void;
  onApprove: (confirmed: boolean) => void;
  onChoose: (model: string) => void;
  onRetry: () => void;
}) {
  const klass = ["stage", current ? "current" : "", stage.status].join(" ");
  const report = stage.report;
  return (
    <section className={klass}>
      <div className="stage-head">
        <div>
          <h3>{stage.title}</h3>
          <p>{stage.summary}</p>
        </div>
        <span className={`pill ${stage.status}`}>{stage.status}</span>
      </div>
      {report ? <ReportView stageId={stage.id} report={report} /> : null}
      {current && (canNext || canRework || canApprove || canChoose || canRetry) ? (
        <div className="prompt-box">
          {canChoose && choices.length > 0 ? (
            <>
              <label>Pick one model to train</label>
              <div className="choices">
                {choices.map((item) => {
                  const selected = chosen[0] === item.id;
                  return (
                    <button
                      type="button"
                      key={item.id}
                      className={selected ? "choice on" : "choice"}
                      disabled={busy}
                      onClick={() => onChoose(item.id)}
                    >
                      <strong>{item.title}</strong>
                      <span>{item.reason}</span>
                    </button>
                  );
                })}
              </div>
            </>
          ) : null}
          {canRetry ? (
            <div className="actions">
              <button className="btn btn-primary" disabled={busy} onClick={onRetry}>
                Retry this stage
              </button>
            </div>
          ) : canApprove ? (
            <div className="actions">
              <button className="btn btn-primary" disabled={busy} onClick={() => onApprove(true)}>
                Approve training
              </button>
              <button className="btn btn-ghost" disabled={busy} onClick={() => onApprove(false)}>
                Reject
              </button>
            </div>
          ) : (
            <>
              <label>Note to this stage</label>
              <textarea
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                placeholder="What should this stage reconsider?"
              />
              <div className="actions">
                <button
                  className="btn btn-ghost"
                  disabled={busy || !prompt.trim()}
                  onClick={onRework}
                >
                  Send note
                </button>
                <button className="btn btn-primary" disabled={busy || !canNext} onClick={onNext}>
                  Continue
                </button>
              </div>
            </>
          )}
        </div>
      ) : null}
    </section>
  );
}
