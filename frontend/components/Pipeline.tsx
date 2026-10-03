"use client";

import Chart from "./Chart";
import { Comparison, Report } from "./Parts";
import { FeaturesReview, PlanReview, PromoteReview, type Act } from "./Reviews";
import { fmt, Info, MetricName, num, useWords } from "@/lib/words";
import type { FeatureDetails, Leaderboard, Pipeline as PipelineView, Profile, Stage } from "@/lib/types";

const STATE: Record<string, [string, string]> = {
  waiting: ["", "Waiting"],
  working: ["●", "Working"],
  reviewing: ["●", "Skeptic is reviewing"],
  your_review: ["✋", "Your turn"],
  done: ["✓", "Done"],
  kept: ["✓", "Kept for later"],
  discarded: ["✕", "Discarded"],
};
export const WORKING: Record<string, string> = {
  profile: "Analyst is reading the data",
  features: "Engineer is building features",
  model: "Engineer is training candidates",
};
const SPLIT: Record<string, string> = { train: "Learn from", valid: "Compare on", test: "Final test" };

function signalHelp(positive: string) {
  return `How well this column alone tells ${positive}s apart: 0.5 is no signal, 1 is perfect. Close to 1 usually means it was filled in after the fact.`;
}

function ProfileDetails({ p }: { p?: Profile | null }) {
  const { record, positive } = useWords();
  if (!p?.column_profiles) return null;
  return (
    <>
      <div className="kpis">
        <div className="kpi">
          <b>{num(p.rows)}</b>
          <span>{record}s</span>
        </div>
        <div className="kpi">
          <b>{num(p.columns)}</b>
          <span>columns</span>
        </div>
        <div className="kpi">
          <b>{num(100 * p.target.positive_rate, 1)}%</b>
          <span>ended in a {positive}</span>
        </div>
        <div className="kpi">
          <b>{num(p.duplicate_rows)}</b>
          <span>exact duplicates</span>
        </div>
      </div>
      {(p.charts || []).map((c, i) => (
        <Chart key={i} spec={c} />
      ))}
      <div className="scroll">
        <table className="grid">
          <thead>
            <tr>
              <th>Column</th>
              <th>Kind</th>
              <th className="num">Missing</th>
              <th className="num">Distinct values</th>
              <th className="num">
                Predictive power
                <Info text={signalHelp(positive)} />
              </th>
              <th>Range / common values</th>
            </tr>
          </thead>
          <tbody>
            {p.column_profiles.map((c) => (
              <tr key={c.name}>
                <td>
                  <code>{c.name}</code>
                </td>
                <td>{c.kind}</td>
                <td className="num">{num(c.missing_pct, 1)}%</td>
                <td className="num">{num(c.unique)}</td>
                <td className="num">{num(c.signal_auc)}</td>
                <td className="muted">{c.stats ? `${num(c.stats.min, 2)} … ${num(c.stats.max, 2)}` : (c.top || []).map(([v]) => v).join(", ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function FeatureDetailsView({ d }: { d?: FeatureDetails | null }) {
  const { positive } = useWords();
  if (!d) return null;
  const tables = d.tables || {};
  const removed = Object.entries(d.rows_removed || {});
  const excluded = Object.entries(d.excluded_columns || {});
  return (
    <>
      <div className="kpis">
        {["train", "valid", "test"].map((s) =>
          tables[s] ? (
            <div className="kpi" key={s}>
              <b>{num(tables[s].rows)}</b>
              <span>
                {SPLIT[s]} · {num(100 * tables[s].outcome_rate, 0)}% {positive}s
              </span>
            </div>
          ) : null,
        )}
        {d.store && (
          <div className="kpi">
            <b>
              {d.store.view} {d.store.version}
            </b>
            <span>saved in the feature store</span>
          </div>
        )}
      </div>
      {d.split && (
        <div className="small">
          <b>How the data is split:</b> {d.split.reason || d.split.method}
        </div>
      )}
      {removed.length > 0 && (
        <div className="small">
          <b>Rows removed:</b> {removed.map(([k, v]) => `${k} (${num(v)})`).join("; ")}
        </div>
      )}
      {excluded.length > 0 && (
        <div className="small">
          <b>Columns left out:</b>{" "}
          {excluded.map(([k, v], i) => (
            <span key={k}>
              {i > 0 && "; "}
              <code>{k}</code> {v}
            </span>
          ))}
        </div>
      )}
      <div className="scroll">
        <table className="grid">
          <thead>
            <tr>
              <th>Feature</th>
              <th>What it is</th>
              <th>Built from</th>
              <th className="num">Missing</th>
              <th className="num">
                Predictive power
                <Info text={signalHelp(positive)} />
              </th>
              <th />
            </tr>
          </thead>
          <tbody>
            {d.features.map((f) => (
              <tr key={f.name}>
                <td>
                  <code>{f.name}</code>
                </td>
                <td>{f.description}</td>
                <td className="muted">{(f.source || []).join(", ")}</td>
                <td className="num">{num(f.null_pct, 1)}%</td>
                <td className="num">{num(f.signal_auc)}</td>
                <td>{f.known_at_prediction === "yes" ? null : <span className="pill warn">may be known only later</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

/** Every score, for whoever wants the detail. The promote review keeps to three. */
function AllScores({ lb }: { lb?: Leaderboard | null }) {
  const { metrics, record } = useWords();
  if (!lb) return null;
  const keys = Object.keys(metrics);
  const rows = [...lb.rows, { model: "baseline", label: "Doing nothing", ...lb.baseline }];
  const cutoff = `A ${record} is flagged when its score reaches this. It is set to keep the loss lowest.`;
  const table = (split: "valid" | "test", title: string) => (
    <>
      <div className="small muted">{title}</div>
      <div className="scroll">
        <table className="grid">
          <thead>
            <tr>
              <th>Model</th>
              {keys.map((k) => (
                <th className="num" key={k}>
                  <MetricName id={k} />
                  {k === lb.metric ? " ★" : ""}
                </th>
              ))}
              <th className="num">
                Cut-off
                <Info text={cutoff} />
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.model} className={r.model === lb.best ? "best" : ""}>
                <td className="nowrap">{r.label}</td>
                {keys.map((k) => (
                  <td className="num" key={k}>
                    {fmt(metrics, k, r[split][k])}
                  </td>
                ))}
                <td className="num">{num(r.threshold, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
  const compared =
    lb.chosen_on === "valid"
      ? "Compared on held-out rows (picks the winner and the cut-off)"
      : `Compared by ${lb.chosen_on.slice(2)}-fold cross-validation (picks the winner and the cut-off)`;
  return (
    <>
      <Comparison lb={lb} />
      {table("valid", compared)}
      {table("test", "Final test on rows no model saw")}
      <div className="small muted">{lb.protocol}</div>
    </>
  );
}

function StepBody({ stage }: { stage: Stage }) {
  return (
    <>
      {stage.headline && <p className="headline">{stage.headline}</p>}
      <Report items={stage.findings}>
        {stage.key === "profile" &&
          (stage.charts || []).map((c, i) => <Chart key={i} spec={c} />)}
        {stage.key === "profile" && stage.details && <ProfileDetails p={stage.details as Profile} />}
        {stage.key === "features" && stage.details && <FeatureDetailsView d={stage.details as FeatureDetails} />}
        {stage.key === "model" && stage.details && <AllScores lb={stage.details as Leaderboard} />}
      </Report>
      {stage.key === "model" && !!stage.problems?.length && stage.state !== "working" && <div className="problems">{stage.problems.join("; ")}</div>}
    </>
  );
}

export default function Pipeline({ pipeline, act }: { pipeline: PipelineView; act: Act }) {
  return (
    <div>
      {pipeline.stages.map((s, i) => {
        let [icon, text] = STATE[s.state] || STATE.waiting;
        if (s.state === "working") text = WORKING[s.key] || text;
        if (s.state === "waiting") icon = String(i + 1);
        if (s.decided_by === "team" && ["done", "kept", "discarded"].includes(s.state)) text += " · decided by the team";
        if (s.reused_from && s.state === "done") text += ` · reused from ${s.reused_from} (no feedback, same data)`;
        if (s.rounds) text += ` · ${s.rounds} round${s.rounds > 1 ? "s" : ""} of fixes`;
        const reviewing = s.state === "your_review";
        return (
          <div className={`step ${s.state}`} key={s.key}>
            <div className="icon">{icon}</div>
            <div>
              <div className="title">
                {s.label}
                <span className="who">{s.agent}</span>
              </div>
              <div className="state">{text}</div>
              <StepBody stage={s} />
            </div>
            {reviewing && (
              <div className="review">
                {pipeline.why && <div className="why">✋ {pipeline.why}</div>}
                {s.key === "features" && <FeaturesReview stage={s} act={act} />}
                {s.key === "plan" && <PlanReview stage={s} config={pipeline.config} store={(pipeline.stages[1].details as FeatureDetails | null)?.store} act={act} />}
                {s.key === "promote" && <PromoteReview stage={s} act={act} />}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
