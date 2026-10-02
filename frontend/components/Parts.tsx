"use client";

// Small pieces the pipeline stages, reviews and side panels share.

import { Children, type ReactNode } from "react";
import Chart from "./Chart";
import { Inline } from "./Markdown";
import { chartValue, useWords } from "@/lib/words";
import type { Leaderboard } from "@/lib/types";

export function More({ label, children }: { label: string; children: ReactNode }) {
  if (!children) return null;
  return (
    <details className="more">
      <summary>{label}</summary>
      <div className="detail-body">{children}</div>
    </details>
  );
}

export function Findings({ items, className = "receipt" }: { items?: string[]; className?: string }) {
  if (!items?.length) return null;
  return (
    <ul className={className}>
      {items.map((f, i) => (
        <li key={i}>
          <Inline text={f} />
        </li>
      ))}
    </ul>
  );
}

export const POINTS = 3; // what each team member shows up front

/** A team member's report: its first three points, everything else (more points,
 * charts, tables) folded under "Full report". */
export function Report({ items, className, children }: { items?: string[]; className?: string; children?: ReactNode }) {
  const top = (items || []).slice(0, POINTS);
  const rest = (items || []).slice(POINTS);
  const more = rest.length > 0 || Children.toArray(children).length > 0;
  return (
    <>
      <Findings items={top} className={className} />
      {more && (
        <More label="Full report">
          <Findings items={rest} className={className} />
          {children}
        </More>
      )}
    </>
  );
}

export function Comparison({ lb }: { lb: Leaderboard }) {
  const { metrics } = useWords();
  if (lb.rows.length < 2) return null;
  return (
    <Chart
      spec={{
        type: "hbar",
        title: `${metrics[lb.metric]?.plain || lb.metric} by model`,
        categories: lb.rows.map((r) => r.label),
        series: [
          { name: "compared on", values: lb.rows.map((r) => chartValue(metrics, lb.metric, r.valid[lb.metric])) },
          { name: "final test", values: lb.rows.map((r) => chartValue(metrics, lb.metric, r.test[lb.metric])) },
        ],
      }}
    />
  );
}
