"use client";

// Inline SVG charts: one hue per series in fixed order, round axis ticks, hover tooltips.

import { useRef, useState, type MouseEvent, type ReactNode } from "react";
import type { ChartSpec } from "@/lib/types";

const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];
type Row = [string, number | null, number];
type Tip = { x: number; y: number; label: string; rows: Row[] } | null;
type Hover = (e: MouseEvent, label: string, rows: Row[]) => void;

/** Round tick steps (1, 2, 2.5, 5 × 10^k) so axes read 0, 25, 50 rather than 23.57. */
function ticks(lo: number, hi: number, n = 4): number[] {
  const raw = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((x) => x >= raw) ?? raw;
  const out: number[] = [];
  for (let t = Math.ceil(lo / step - 1e-9) * step; t <= hi + 1e-9; t += step) out.push(+t.toFixed(10));
  return out;
}

function extent(spec: ChartSpec): [number, number] {
  const values = spec.series.flatMap((s) => s.values).filter((v): v is number => v !== null);
  let [lo, hi] = spec.domain || [Math.min(0, ...values), Math.max(...values)];
  if (hi === lo) hi = lo + 1;
  if (!spec.domain) {
    const t = ticks(lo, hi);
    const step = t[1] - t[0];
    if (step > 0) hi = Math.ceil(hi / step - 1e-9) * step;
  }
  return [lo, hi];
}

const rowsAt = (spec: ChartSpec, i: number): Row[] => spec.series.map((s, j) => [s.name, s.values[i], j]);
const round = (v: number) => +v.toFixed(2);

function CategoryLabel({ x, y, text, tilt }: { x: number; y: number; text: string; tilt: boolean }) {
  const label = text.slice(0, 14);
  return tilt ? (
    <text transform={`translate(${x},${y}) rotate(-40)`} textAnchor="end">
      {label}
    </text>
  ) : (
    <text x={x} y={y} textAnchor="middle">
      {label}
    </text>
  );
}

function HBar({ spec, width, hover }: { spec: ChartSpec; width: number; hover: Hover }) {
  const cats = spec.categories;
  const k = spec.series.length;
  const [lo, hi] = extent(spec);
  const left = Math.min(180, 12 + 6.2 * Math.max(...cats.map((c) => String(c).length)));
  const band = k > 1 ? 10 * k + 8 : 20;
  const H = cats.length * band + 22;
  const x = (v: number) => left + ((width - left - 44) * (v - lo)) / (hi - lo);
  return (
    <svg viewBox={`0 0 ${width} ${H}`} role="img" aria-label={spec.title}>
      {ticks(lo, hi).map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={0} y2={H - 18} stroke="var(--grid)" />
          <text x={x(t)} y={H - 4} textAnchor="middle">
            {round(t)}
          </text>
        </g>
      ))}
      {cats.map((c, i) => {
        const y0 = i * band;
        return (
          <g key={i}>
            <text x={left - 8} y={y0 + band / 2 + 4} textAnchor="end">
              {String(c).slice(0, 28)}
            </text>
            {spec.series.map((s, j) => {
              const v = s.values[i];
              if (v === null) return null;
              const h = k > 1 ? 8 : 14;
              const y = y0 + (band - (k > 1 ? 10 * k - 2 : h)) / 2 + (k > 1 ? j * 10 : 0);
              const w = Math.max(1, x(v) - x(lo));
              return (
                <g key={j}>
                  <rect className="mark" x={x(lo)} y={y} width={w} height={h} rx={3} fill={SERIES[j]} />
                  {k === 1 && (
                    <text className="value" x={x(lo) + w + 5} y={y + h - 3}>
                      {+v.toFixed(3)}
                    </text>
                  )}
                </g>
              );
            })}
            <rect className="hit" x={0} y={y0} width={width} height={band} onMouseMove={(e) => hover(e, c, rowsAt(spec, i))} />
          </g>
        );
      })}
    </svg>
  );
}

function Columns({ spec, width, hover, stacked }: { spec: ChartSpec; width: number; hover: Hover; stacked: boolean }) {
  const cats = spec.categories;
  const k = spec.series.length;
  let [lo, hi] = extent(spec);
  if (stacked) {
    const top = Math.max(...cats.map((_, i) => spec.series.reduce((a, s) => a + (s.values[i] || 0), 0))) || 1;
    const t = ticks(0, top);
    [lo, hi] = [0, Math.max(top, t[t.length - 1])];
  }
  const H = 220;
  const left = 40;
  const tilt = cats.length > 8;
  const bottom = tilt ? 56 : 24;
  const band = (width - left) / cats.length;
  const y = (v: number) => 8 + (H - bottom - 8) * (1 - (v - lo) / (hi - lo));
  const bw = stacked ? Math.max(4, Math.min(48, band - 10)) : Math.max(2, Math.min(28, (band - 4) / k - 2));
  const zero = y(Math.max(lo, 0));
  return (
    <svg viewBox={`0 0 ${width} ${H}`} role="img" aria-label={spec.title}>
      {ticks(lo, hi).map((t) => (
        <g key={t}>
          <line x1={left} x2={width} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
          <text x={left - 6} y={y(t) + 4} textAnchor="end">
            {round(t)}
          </text>
        </g>
      ))}
      {cats.map((c, i) => {
        const cx = left + band * i + band / 2;
        let base = 0;
        const marks: ReactNode[] = spec.series.map((s, j) => {
          const v = s.values[i];
          if (v === null || v === undefined) return null;
          if (stacked) {
            const rect = (
              <rect key={j} className="mark" x={cx - bw / 2} y={y(base + v)} width={bw} height={Math.max(0, y(base) - y(base + v) - 1)} fill={SERIES[j]} />
            );
            base += v;
            return rect;
          }
          const bx = cx - (k * (bw + 2)) / 2 + j * (bw + 2);
          return <rect key={j} className="mark" x={bx} y={Math.min(y(v), zero)} width={bw} height={Math.max(1, Math.abs(y(v) - zero))} rx={3} fill={SERIES[j]} />;
        });
        return (
          <g key={i}>
            {marks}
            <CategoryLabel x={cx} y={tilt ? H - bottom + 12 : H - 6} text={String(c)} tilt={tilt} />
            <rect className="hit" x={cx - band / 2} y={0} width={band} height={H - bottom} onMouseMove={(e) => hover(e, c, rowsAt(spec, i))} />
          </g>
        );
      })}
    </svg>
  );
}

function Line({ spec, width, hover }: { spec: ChartSpec; width: number; hover: Hover }) {
  const cats = spec.categories;
  const [lo, hi] = extent(spec);
  const H = 220;
  const left = 40;
  const tilt = cats.length > 10;
  const bottom = tilt ? 56 : 24;
  const step = (width - left - 10) / Math.max(1, cats.length - 1);
  const x = (i: number) => left + i * step;
  const y = (v: number) => 8 + (H - bottom - 8) * (1 - (v - lo) / (hi - lo));
  const every = Math.ceil(cats.length / 10);
  return (
    <svg viewBox={`0 0 ${width} ${H}`} role="img" aria-label={spec.title}>
      {ticks(lo, hi).map((t) => (
        <g key={t}>
          <line x1={left} x2={width} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
          <text x={left - 6} y={y(t) + 4} textAnchor="end">
            {round(t)}
          </text>
        </g>
      ))}
      {spec.series.map((s, j) => (
        <polyline
          key={j}
          points={s.values.map((v, i) => (v === null ? null : `${x(i)},${y(v)}`)).filter(Boolean).join(" ")}
          fill="none"
          stroke={SERIES[j]}
          strokeWidth={2}
          strokeLinejoin="round"
        />
      ))}
      {cats.map((c, i) => (
        <g key={i}>
          {i % every === 0 && <CategoryLabel x={x(i)} y={tilt ? H - bottom + 12 : H - 6} text={String(c)} tilt={tilt} />}
          <rect className="hit" x={x(i) - step / 2} y={0} width={step} height={H - bottom} onMouseMove={(e) => hover(e, c, rowsAt(spec, i))} />
        </g>
      ))}
    </svg>
  );
}

export default function Chart({ spec, width = 560 }: { spec?: ChartSpec | null; width?: number }) {
  const box = useRef<HTMLDivElement>(null);
  const [tip, setTip] = useState<Tip>(null);
  if (!spec || !spec.categories?.length) return null;
  const hover: Hover = (e, label, rows) => {
    const r = box.current?.getBoundingClientRect();
    if (r) setTip({ x: e.clientX - r.left + 12, y: e.clientY - r.top + 12, label, rows });
  };
  const body =
    spec.type === "hbar" ? (
      <HBar spec={spec} width={width} hover={hover} />
    ) : spec.type === "line" ? (
      <Line spec={spec} width={width} hover={hover} />
    ) : (
      <Columns spec={spec} width={width} hover={hover} stacked={spec.type === "stacked"} />
    );
  return (
    <div className="chart" ref={box} onMouseLeave={() => setTip(null)}>
      <div className="ctitle">{spec.title}</div>
      {body}
      {spec.series.length > 1 && (
        <div className="legend">
          {spec.series.map((s, i) => (
            <span key={i}>
              <i style={{ background: SERIES[i] }} />
              {s.name}
            </span>
          ))}
        </div>
      )}
      {tip && (
        <div className="tip" style={{ left: tip.x, top: tip.y }}>
          <div className="muted">{tip.label}</div>
          {tip.rows.map(([name, v, j]) => (
            <div key={j}>
              <i className="key" style={{ background: SERIES[j] }} />
              <b>{v === null ? "—" : +(+v).toFixed(4)}</b> {name}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
