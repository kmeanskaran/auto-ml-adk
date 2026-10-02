"use client";

// The run's trace (runs/<run>/logs/activity.log): one line per agent step, tool call
// and decision. Stays pinned to the bottom unless the reader has scrolled up.

import { useEffect, useRef } from "react";

export default function Activity({ lines, run }: { lines: string[]; run: string }) {
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
