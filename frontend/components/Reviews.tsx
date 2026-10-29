"use client";

// The three places the team may ask the human: features, training plan, go-live.

import { useState } from "react";
import { Comparison, More, Report } from "./Parts";
import { Inline } from "./Markdown";
import { fmt, Info, MetricName, useWords, Verdict } from "@/lib/words";
import type { Answer, Config, Leaderboard, Review, Stage } from "@/lib/types";

export type Act = (answer: Answer) => Promise<void>;

function Skeptic({ review, subject }: { review?: Review | null; subject: string }) {
  if (!review?.verdict) return <div className="muted small">The skeptic did not send a review.</div>;
  const good = review.verdict === "pass";
  return (
    <div className="skeptic">
      <div className="says">
        <span className={`pill ${good ? "good" : "warn"}`}>{good ? "✓ Sound" : "⚠ Concerns"}</span>
        <span>
          The skeptic {good ? `is happy with ${subject}.` : `would change ${subject} first.`}
        </span>
      </div>
      <Report items={review.findings} className="plain" />
    </div>
  );
}

/** The skeptic's changes, most impactful first, ticked by default: the team's next move. */
function Suggestions({ review, picked, setPicked }: { review?: Review | null; picked: number[]; setPicked: (p: number[]) => void }) {
  const items = review?.recommendations || [];
  if (!items.length) return null;
  const toggle = (i: number) => setPicked(picked.includes(i) ? picked.filter((x) => x !== i) : [...picked, i]);
  return (
    <ol className="changes">
      {items.map((text, i) => (
        <li key={i}>
          <label className="check">
            <input type="checkbox" checked={picked.includes(i)} onChange={() => toggle(i)} />
            <span>
              {i === 0 && <span className="pill warn">biggest impact</span>} <Inline text={text} />
            </span>
          </label>
        </li>
      ))}
    </ol>
  );
}

function Note({ text, setText, placeholder }: { text: string; setText: (t: string) => void; placeholder: string }) {
  return (
    <details className="more note" open={!!text}>
      <summary>Add your own instruction</summary>
      <textarea value={text} onChange={(e) => setText(e.target.value)} placeholder={placeholder} />
    </details>
  );
}

const all = (review?: Review | null) => (review?.recommendations || []).map((_, i) => i);

/** Sending work back needs something to send: a ticked suggestion or the human's words. */
function sendsBack(answer: Answer) {
  if (!answer.text && !answer.apply?.length) {
    window.alert("Tick a suggestion or write what should change.");
    return false;
  }
  return true;
}

const discard = (act: Act) => () => window.confirm("Discard this run?") && act({ choice: "discard" });

export function FeaturesReview({ stage, act }: { stage: Stage; act: Act }) {
  const [picked, setPicked] = useState<number[]>(all(stage.review));
  const [text, setText] = useState("");
  const problems = stage.problems || [];
  const answer = (choice: string): Answer => ({ choice, text: text.trim(), apply: picked });
  const changes = picked.length + (text.trim() ? 1 : 0);
  return (
    <>
      <Skeptic review={stage.review} subject="these features" />
      <Suggestions review={stage.review} picked={picked} setPicked={setPicked} />
      <Note text={text} setText={setText} placeholder="Anything else the engineer should add, change or remove?" />
      {problems.length > 0 && <div className="problems">The hand-over is incomplete: {problems.join("; ")}. Send it back first.</div>}
      <div className="actions">
        {changes > 0 && (
          <button className="primary" onClick={() => sendsBack(answer("feedback")) && act(answer("feedback"))}>
            Improve: apply {changes} change{changes > 1 ? "s" : ""}
          </button>
        )}
        <button className={changes === 0 ? "primary" : ""} disabled={problems.length > 0} onClick={() => act(answer("continue"))}>
          Approve features
        </button>
        <span className="spacer" />
        <button className="link" onClick={discard(act)}>
          Discard run
        </button>
      </div>
    </>
  );
}

export function PlanReview({ stage, config, store, act }: { stage: Stage; config: Config; store?: { view: string; version: string }; act: Act }) {
  const { metrics, record } = useWords();
  const proposal = stage.proposal || {};
  const [models, setModels] = useState<string[]>(proposal.models || []);
  const [metric, setMetric] = useState(proposal.metric || "cost_per_1000");
  const [text, setText] = useState("");
  const toggle = (key: string) => setModels(models.includes(key) ? models.filter((m) => m !== key) : [...models, key]);
  const train = () => {
    if (!models.length) return window.alert("Pick at least one model.");
    act({ choice: "train", models, metric, text: text.trim() });
  };
  return (
    <>
      <dl className="facts">
        <dt>Goal</dt>
        <dd>{config.goal || `predict ${config.target}`}</dd>
        <dt>Features</dt>
        <dd>{store ? `${store.view} ${store.version} (approved)` : config.feature_view}</dd>
        <dt>Cut-off</dt>
        <dd>
          set by the mistake costs
          <Info text={`A ${record} is flagged when its score reaches this. It comes from the mistake costs, not from the data.`} />
        </dd>
        {proposal.reason && (
          <>
            <dt>Engineer suggests</dt>
            <dd>{proposal.reason}</dd>
          </>
        )}
      </dl>
      <h4>Which models should compete?</h4>
      <div className="chips">
        {(stage.options?.models || []).map((m) => (
          <label className={`chip ${models.includes(m.key) ? "on" : ""}`} key={m.key}>
            <input type="checkbox" checked={models.includes(m.key)} onChange={() => toggle(m.key)} />
            {m.label}
            {(proposal.models || []).includes(m.key) && <span className="muted small">suggested</span>}
          </label>
        ))}
      </div>
      <h4>How should we pick the winner?</h4>
      <select value={metric} onChange={(e) => setMetric(e.target.value)}>
        {Object.entries(metrics).map(([k, m]) => (
          <option value={k} key={k}>
            {m.plain}
            {k === proposal.metric ? " · suggested" : ""}
          </option>
        ))}
      </select>
      <div className="small muted">{metrics[metric]?.about}</div>
      <textarea value={text} onChange={(e) => setText(e.target.value)} placeholder="Notes for the engineer (optional), e.g. “keep the models simple”" />
      <div className="actions">
        <button className="primary" onClick={train}>
          Train these models
        </button>
        <span className="spacer" />
        <button className="link" onClick={discard(act)}>
          Discard run
        </button>
      </div>
    </>
  );
}

/** The decision table: each model's score on the plan's metric (★, the business one by
 *  default) on the test set, with a bar so the gaps show at a glance. */
function Candidates({ lb, pick, setPick }: { lb: Leaderboard; pick: string; setPick: (m: string) => void }) {
  const { metrics } = useWords();
  const values = lb.rows.map((r) => r.test[lb.metric] ?? 0);
  const top = Math.max(...values, 1e-9);
  return (
    <div className="scroll">
      <table className="grid">
        <thead>
          <tr>
            <th />
            <th>Model</th>
            <th className="num">
              <MetricName id={lb.metric} short /> ★
            </th>
          </tr>
        </thead>
        <tbody>
          {lb.rows.map((r) => (
            <tr key={r.model} className={`clickable ${r.model === pick ? "picked" : ""}`} onClick={() => setPick(r.model)}>
              <td>
                <input type="radio" name="pick" checked={r.model === pick} onChange={() => setPick(r.model)} />
              </td>
              <td className="nowrap">
                {r.label}
                {r.model === lb.best && (
                  <>
                    {" "}
                    <span className="pill good">winner</span>
                  </>
                )}
              </td>
              <td className="num">
                <span className="signal wide">
                  <span className="bar">
                    <i style={{ width: `${Math.round((100 * (r.test[lb.metric] ?? 0)) / top)}%` }} />
                  </span>
                  {fmt(metrics, lb.metric, r.test[lb.metric])}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function PromoteReview({ stage, act }: { stage: Stage; act: Act }) {
  const review = stage.review || {};
  const lb = stage.leaderboard;
  const [pick, setPick] = useState(lb?.best || "");
  const [picked, setPicked] = useState<number[]>(all(review));
  const [text, setText] = useState("");
  const back = (choice: string) => {
    const answer = { choice, text: text.trim(), apply: picked };
    if (sendsBack(answer)) act(answer);
  };
  const changes = picked.length + (text.trim() ? 1 : 0);
  const improve = (
    <div className="improve">
      <h4>Or improve it first</h4>
      <Suggestions review={review} picked={picked} setPicked={setPicked} />
      <Note text={text} setText={setText} placeholder="What should change?" />
      <div className="actions">
        <button disabled={!changes} onClick={() => back("retrain")}>
          Retrain with {changes || "the"} change{changes === 1 ? "" : "s"}
        </button>
        <button disabled={!changes} onClick={() => back("features")}>
          Rework the features
        </button>
        <button onClick={() => act({ choice: "replan" })}>Change models or scoring</button>
      </div>
    </div>
  );
  // Training can finish with no evaluated candidate: nothing to put live, only send back.
  const chosen = lb?.rows.find((r) => r.model === pick) || lb?.rows[0];
  if (!lb || !chosen)
    return (
      <>
        <div className="muted">No model could be scored, so there is nothing to put live. Send it back.</div>
        <Skeptic review={review} subject="the models" />
        {improve}
        <div className="actions">
          <span className="spacer" />
          <button className="link" onClick={discard(act)}>
            Discard run
          </button>
        </div>
      </>
    );
  const live = stage.live;
  return (
    <>
      <div className="compare-live">
        <div className="pick-verdict">
          <div className="small muted">Candidate</div>
          <div className="name">{chosen.label}</div>
          <div className="small">
            <Verdict test={chosen.test} />
          </div>
        </div>
        <div className="pick-verdict quiet">
          <div className="small muted">Live now</div>
          {live ? (
            <>
              <div className="name">
                {live.version} · {live.model}
              </div>
              <div className="small">
                <Verdict test={live.test} />
              </div>
            </>
          ) : (
            <div className="name muted">nothing yet</div>
          )}
        </div>
      </div>
      {lb.warnings.length > 0 && (
        <ul className="warnings">
          {lb.warnings.map((w, i) => (
            <li key={i}>
              <Inline text={w} />
            </li>
          ))}
        </ul>
      )}
      <Skeptic review={review} subject="the models" />
      <Candidates lb={lb} pick={chosen.model} setPick={setPick} />
      <More label="Compare the models on a chart">
        <Comparison lb={lb} />
      </More>
      <div className="actions">
        <button className="primary" onClick={() => window.confirm(`Put ${chosen.label} live? /predict will start using it.`) && act({ choice: "promote", model: chosen.model })}>
          Put {chosen.label} live
        </button>
        <button onClick={() => act({ choice: "keep", model: chosen.model })}>Keep for later</button>
        <span className="spacer" />
        <button className="link" onClick={discard(act)}>
          Discard run
        </button>
      </div>
      {improve}
    </>
  );
}
