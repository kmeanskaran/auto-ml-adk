"use client";

// Finished runs (runs/index.jsonl): what each tried, scored and decided. The next run's
// team reads the same history as hypotheses to test.

import { fmt, tokens, useWords } from "@/lib/words";
import type { RunCard } from "@/lib/types";

export default function History({ runs }: { runs: RunCard[] }) {
  const { metrics } = useWords();
  if (!runs.length) return <div className="empty">No finished runs yet. Each run that ends is recorded here, and the next run rethinks its code with your feedback.</div>;
  return (
    <div className="scroll">
      <table className="grid">
        <thead>
          <tr>
            <th>Run</th>
            <th>Features</th>
            <th>Best model</th>
            <th className="num">Final test</th>
            <th>Ended</th>
            <th className="num">Decisions</th>
            <th className="num" title="Time from the first step to the last, model tokens, and the share served from Gemini's prompt cache">Cost</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.run}>
              <td>
                <b className="nowrap">{r.run.replace(/^run-/, "")}</b>
                {r.feedback && <div className="small muted" title={r.feedback}>“{r.feedback.length > 80 ? `${r.feedback.slice(0, 80)}…` : r.feedback}”</div>}
              </td>
              <td className="nowrap">{r.feature_view ? `${r.feature_view} · ${r.features}` : "—"}</td>
              <td className="nowrap">{r.best || "—"}</td>
              <td className="num nowrap">
                {r.metric && r.test_score != null ? (
                  <>
                    {fmt(metrics, r.metric, r.test_score)} <span className="muted">{metrics[r.metric]?.label || r.metric}</span>
                  </>
                ) : (
                  "—"
                )}
              </td>
              <td>
                {r.version && <span className="pill good">{r.version}</span>} {r.outcome}
                {r.warnings > 0 && <span className="pill warn">{r.warnings} warning{r.warnings > 1 ? "s" : ""}</span>}
              </td>
              <td className="num nowrap">
                {r.decisions}
                {r.human_overrides > 0 && <span className="muted"> · {r.human_overrides} yours</span>}
              </td>
              <td className="num nowrap">
                {r.usage ? (
                  <>
                    {r.usage.minutes} min · {tokens(r.usage.tokens)}
                    <div className="small muted">
                      {r.usage.cached_pct}% cached · {r.usage.tool_calls} tool calls
                    </div>
                  </>
                ) : (
                  "—"
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
