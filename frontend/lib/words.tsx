"use client";

// Business words in front, technical ones behind an ⓘ: the metric texts and the
// config's words ("application", "approval") come from the view.

import { createContext, useContext } from "react";
import type { MetricInfo, Scores } from "./types";

export type Words = {
  metrics: Record<string, MetricInfo>;
  record: string;
  positive: string;
};

export const WordsContext = createContext<Words>({ metrics: {}, record: "record", positive: "positive" });
export const useWords = () => useContext(WordsContext);

export function num(v: unknown, digits = 3): string {
  if (v === null || v === undefined) return "—";
  if (typeof v !== "number") return String(v);
  if (Number.isInteger(v)) return Math.abs(v) >= 10000 ? v.toLocaleString() : String(v);
  return v.toFixed(digits);
}

export const pct = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${Math.round(v * 100)}%`);

export function fmt(metrics: Record<string, MetricInfo>, key: string, v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  const kind = metrics[key]?.kind;
  return kind === "rate" ? pct(v) : kind === "cost" ? Math.round(v).toLocaleString() : num(v);
}

/** A chart shows values the way the tables do: 27 for 27%, 268 for a loss of 267.7. */
export function chartValue(metrics: Record<string, MetricInfo>, key: string, v: number | null | undefined) {
  if (v === null || v === undefined) return null;
  const kind = metrics[key]?.kind;
  return kind === "rate" ? Math.round(v * 100) : kind === "cost" ? Math.round(v) : v;
}

export function Info({ text }: { text?: string }) {
  if (!text) return null;
  return (
    <span className="info" title={text} tabIndex={0}>
      ⓘ
    </span>
  );
}

const SHORT: Record<string, string> = { recall: "Caught", precision: "Right when flagged", cost_per_1000: "Loss per 1k" };

export function MetricName({ id, short = false }: { id: string; short?: boolean }) {
  const { metrics } = useWords();
  const m = metrics[id];
  return (
    <>
      {(short && SHORT[id]) || m?.plain || id}
      <Info text={`${m?.plain || id}: ${m?.about || ""} (${m?.label || id})`} />
    </>
  );
}

/** One sentence a manager can act on: how many it catches, how often it is right, what it costs. */
export function Verdict({ test, baseCost }: { test?: Scores; baseCost?: number | null }) {
  const { metrics, record, positive } = useWords();
  if (!test) return null;
  return (
    <>
      {test.recall != null && (
        <>
          Catches <b>{pct(test.recall)}</b> of {positive}s.{" "}
        </>
      )}
      {test.precision != null && (
        <>
          When it flags a {record}, it is right <b>{pct(test.precision)}</b> of the time.{" "}
        </>
      )}
      {test.cost_per_1000 != null && (
        <>
          Loss: <b>{fmt(metrics, "cost_per_1000", test.cost_per_1000)}</b> per 1,000 {record}s
          {baseCost ? ` (doing nothing: ${fmt(metrics, "cost_per_1000", baseCost)})` : ""}.
        </>
      )}
    </>
  );
}
