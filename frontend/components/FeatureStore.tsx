"use client";

// The feature store: every saved version of each feature view, and its features.
// Training and serving both use a version's build(), so this is what models see.

import { useState } from "react";
import { Info, num, useWords } from "@/lib/words";
import type { FeatureVersion, FeatureView } from "@/lib/types";

const SPLIT: Record<string, string> = { train: "Learn from", valid: "Compare on", test: "Final test" };

function Signal({ auc }: { auc?: number | null }) {
  if (auc === null || auc === undefined) return <span className="muted">—</span>;
  const strength = Math.max(0, Math.min(1, (Math.abs(auc - 0.5) * 2)));
  return (
    <span className="signal" title={`signal AUC ${auc}`}>
      <span className="bar">
        <i style={{ width: `${Math.round(strength * 100)}%` }} />
      </span>
      {num(auc, 2)}
    </span>
  );
}

function Version({ v }: { v: FeatureVersion }) {
  const { positive } = useWords();
  const excluded = Object.entries(v.excluded_columns || {});
  const flagged = v.columns.filter((c) => c.protected.length).length;
  return (
    <>
      <div className="kpis">
        {["train", "valid", "test"].map((s) =>
          v.tables[s] ? (
            <div className="kpi" key={s}>
              <b>{num(v.tables[s].rows)}</b>
              <span>
                {SPLIT[s]} · {num(100 * v.tables[s].outcome_rate, 0)}% {positive}s
              </span>
            </div>
          ) : null,
        )}
        <div className="kpi">
          <b>{v.split || "—"}</b>
          <span>split</span>
        </div>
        <div className="kpi">
          <b>{v.used_by.length ? v.used_by.map((m) => m.version).join(", ") : "none"}</b>
          <span>models built on it</span>
        </div>
      </div>
      {flagged > 0 && (
        <div className="small notice">
          {flagged} feature{flagged > 1 ? "s are" : " is"} built from a protected attribute. The fair-lending check compares those groups before any model goes live.
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
              <th>
                Predictive power
                <Info text={`How well the feature alone tells ${positive}s apart: 0.5 is none, 1 is perfect.`} />
              </th>
              <th />
            </tr>
          </thead>
          <tbody>
            {v.columns.map((c) => (
              <tr key={c.name}>
                <td>
                  <code>{c.name}</code>
                </td>
                <td>{c.description}</td>
                <td className="muted">{c.source.join(", ")}</td>
                <td className="num">{c.null_pct == null ? "—" : `${num(c.null_pct, 1)}%`}</td>
                <td>
                  <Signal auc={c.signal_auc} />
                </td>
                <td className="nowrap">
                  {c.protected.length > 0 && <span className="pill warn">protected: {c.protected.join(", ")}</span>}{" "}
                  {c.known_at_prediction && c.known_at_prediction !== "yes" && <span className="pill warn">may be known only later</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {excluded.length > 0 && (
        <div className="small muted">
          Left out:{" "}
          {excluded.map(([k, why], i) => (
            <span key={k}>
              {i > 0 && "; "}
              <code>{k}</code> {why}
            </span>
          ))}
        </div>
      )}
      <div className="small muted">
        Saved {v.created_at.slice(0, 16).replace("T", " ")} by run {v.run}.
      </div>
    </>
  );
}

export default function FeatureStore({ views }: { views: FeatureView[] }) {
  const [viewName, setViewName] = useState("");
  const [version, setVersion] = useState("");
  if (!views.length) return <div className="empty">No features saved yet. Approved features are saved here as a version when a run continues past the features review.</div>;
  const view = views.find((v) => v.view === viewName) || views[0];
  const shown = view.versions.find((v) => v.version === version) || view.versions.find((v) => v.live) || view.versions[0];
  return (
    <div className="stack">
      <div className="toolbar">
        {views.length > 1 && (
          <select value={view.view} onChange={(e) => (setViewName(e.target.value), setVersion(""))}>
            {views.map((v) => (
              <option key={v.view}>{v.view}</option>
            ))}
          </select>
        )}
        <span className="muted small">{views.length === 1 && <code>{view.view}</code>}</span>
        <div className="segmented">
          {view.versions.map((v) => (
            <button key={v.version} className={v.version === shown.version ? "on" : ""} onClick={() => setVersion(v.version)}>
              {v.version} <span className="muted">· {v.features}</span>
              {v.live && <span className="dot live-dot" title="the live model uses this version" />}
            </button>
          ))}
        </div>
      </div>
      <Version v={shown} />
    </div>
  );
}
