"use client";

// Every kept version, compared on one score: make one live, or remove it.

import { useState } from "react";
import Chart from "./Chart";
import { chartValue, fmt, MetricName, useWords } from "@/lib/words";
import type { VersionCard } from "@/lib/types";

const STATUS: Record<string, [string, string]> = {
  production: ["live", "good"],
  candidate: ["kept", ""],
  archived: ["retired", ""],
};

export default function Library({ versions, onLive, onRemove }: { versions: VersionCard[]; onLive: (v: string) => void; onRemove: (v: string) => void }) {
  const { metrics, record } = useWords();
  const [key, setKey] = useState("cost_per_1000");
  const [choice, setChoice] = useState("");
  if (!versions.length) return <div className="empty">No models yet. Every model the team keeps or puts live is listed here, with its scores.</div>;
  const m = metrics[key];
  const scored = versions.filter((v) => typeof v.test[key] === "number");
  const best = scored.reduce<VersionCard | null>(
    (a, v) => (!a || (m?.higher_is_better ? (v.test[key] as number) > (a.test[key] as number) : (v.test[key] as number) < (a.test[key] as number)) ? v : a),
    null,
  );
  const cols = ["recall", "precision", "cost_per_1000"].filter((k) => k !== key);
  const pick = versions.find((v) => v.version === choice) || best || versions[0];
  const live = pick.status === "production";
  return (
    <div className="stack">
      <div className="toolbar">
        <span className="spacer" />
        <label className="small muted">Compare on</label>
        <select value={key} onChange={(e) => setKey(e.target.value)}>
          {Object.entries(metrics).map(([k, x]) => (
            <option value={k} key={k}>
              {x.plain}
            </option>
          ))}
        </select>
      </div>
      {scored.length > 1 && <Chart
        spec={{
          type: "hbar",
          title: `${m?.plain || key}${m?.kind === "rate" ? " (%)" : ""} by version, final test`,
          categories: scored.map((v) => v.version),
          series: [{ name: m?.plain || key, values: scored.map((v) => chartValue(metrics, key, v.test[key])) }],
        }}
      />}
      <div className="scroll">
        <table className="grid">
          <thead>
            <tr>
              <th />
              <th>Version</th>
              <th>Status</th>
              <th>Model</th>
              <th className="num">
                <MetricName id={key} short /> ★
              </th>
              {cols.map((k) => (
                <th className="num" key={k}>
                  <MetricName id={k} short />
                </th>
              ))}
              <th>Created</th>
            </tr>
          </thead>
          <tbody>
            {versions.map((v) => {
              const [status, tone] = STATUS[v.status] || [v.status, ""];
              return (
                <tr key={v.version} className={`clickable ${v.version === pick.version ? "picked" : ""}`} onClick={() => setChoice(v.version)}>
                  <td>
                    <input type="radio" name="lib" checked={v.version === pick.version} onChange={() => setChoice(v.version)} />
                  </td>
                  <td className="nowrap">
                    <b>{v.version}</b>
                    {best?.version === v.version && (
                      <>
                        {" "}
                        <span className="pill warn">recommended</span>
                      </>
                    )}
                  </td>
                  <td>
                    <span className={`pill ${tone}`}>{status}</span>
                  </td>
                  <td className="nowrap">{v.model || "—"}</td>
                  <td className="num">
                    <b>{fmt(metrics, key, v.test[key])}</b>
                  </td>
                  {cols.map((k) => (
                    <td className="num" key={k}>
                      {fmt(metrics, k, v.test[k])}
                    </td>
                  ))}
                  <td className="muted nowrap">{(v.created_at || "").slice(0, 10)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="small">
        {best && versions.length > 1 && (
          <>
            <b>{best.version}</b> is recommended: it has the {m?.higher_is_better ? "highest" : "lowest"} {(m?.plain || key).toLowerCase()} ({fmt(metrics, key, best.test[key])}).{" "}
          </>
        )}
        <span className="muted">Versions built on different feature sets were tested on different {record}s, so small gaps may not mean much.</span>
      </div>
      <div className="actions">
        <b>{pick.version}</b>
        <button className="primary" disabled={live} onClick={() => window.confirm(`Make ${pick.version} (${pick.model}) live? /predict will start using it.`) && onLive(pick.version)}>
          {live ? "Live now" : "Make live"}
        </button>
        <button disabled={live} onClick={() => window.confirm(`Remove ${pick.version} from the library? It can no longer be used or made live.`) && onRemove(pick.version)}>
          Remove
        </button>
        <span className="spacer" />
        <code>POST /predict/{pick.version}</code>
      </div>
    </div>
  );
}
