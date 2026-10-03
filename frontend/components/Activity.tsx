"use client";

// The run's trace (runs/<run>/logs/activity.log): one line per agent step, tool call
// and decision. Stays pinned to the bottom unless the reader has scrolled up.

import { useEffect, useRef } from "react";

import type { Usage } from "@/lib/types";
import { tokens } from "@/lib/words";

// Who spent the time and the tokens: one row per agent turn, the run's total last.
function Turns({ usage }: { usage: Usage }) {
  const turns = usage.turns || [];
  if (!turns.length) return null;
  const share = (cached: number, all: number) => (all ? `${Math.round((100 * cached) / all)}%` : "—");
  return (
    <table className="grid small">
      <thead>
        <tr>
          <th>Agent</th>
          <th className="num">Time</th>
          <th className="num">Tool calls</th>
          <th className="num">Tokens</th>
          <th className="num" title="Share of the tokens served from Gemini's prompt cache, billed at a discount">Cached</th>
        </tr>
      </thead>
      <tbody>
        {turns.map((t, i) => (
          <tr key={i}>
            <td className="nowrap">{t.agent}</td>
            <td className="num">{(t.seconds / 60).toFixed(1)} min</td>
            <td className="num">{t.tool_calls}</td>
            <td className="num">{tokens(t.tokens)}</td>
            <td className="num">{share(t.cached_tokens, t.tokens)}</td>
          </tr>
        ))}
        <tr>
          <td>
            <b>Total</b> <span className="muted">({usage.minutes} min start to end)</span>
          </td>
          <td className="num">
            <b>{usage.agent_minutes} min</b>
          </td>
          <td className="num">
            <b>{usage.tool_calls}</b>
          </td>
          <td className="num">
            <b>{tokens(usage.tokens)}</b>
          </td>
          <td className="num">
            <b>{usage.cached_pct}%</b>
          </td>
        </tr>
      </tbody>
    </table>
  );
}

export default function Activity({ lines, run, usage }: { lines: string[]; run: string; usage?: Usage | null }) {
  const box = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  useEffect(() => {
    if (box.current && stick.current) box.current.scrollTop = box.current.scrollHeight;
  }, [lines]);
  const onScroll = () => {
    const b = box.current;
    if (b) stick.current = b.scrollTop + b.clientHeight >= b.scrollHeight - 20;
  };
  return (
    <div className="stack">
      {usage && <Turns usage={usage} />}
      {run && <div className="small muted">The full trace is in runs/{run}/logs/: activity.log, trace.jsonl and every version of the code.</div>}
      <div className="log" ref={box} onScroll={onScroll}>
        {lines.length === 0 ? (
          <span className="muted">Every agent step, tool call and decision appears here once a run starts.</span>
        ) : (
          lines.map((line, i) => {
            const [, time = "", agent = "", text = line] = line.match(/^(\S+)\s+(\S+)\s+(.*)$/) || [];
            const tone = /✗/.test(text) ? "bad" : /^◆/.test(text) ? "dec" : "";
            return (
              <div key={i}>
                <span className="t">{time}</span> <span className="a">{agent.padEnd(18)}</span> <span className={tone}>{text}</span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
