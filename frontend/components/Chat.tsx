"use client";

// Questions about the data and the pipeline's results, answered by the analyst agent
// with stats, tables and charts. Ask for "a report on …" for a longer answer.

import { useEffect, useRef, useState } from "react";
import Chart from "./Chart";
import Markdown from "./Markdown";
import { More } from "./Parts";
import type { ChatMessage, View } from "@/lib/types";

const STARTERS = [
  "Which columns matter most for approval, and is each effect real?",
  "Is production traffic different from the data we trained on?",
  "What does the live model rely on?",
  "Do approval rates differ by gender or marriage, in the data and in the model?",
  "A report on income and approval",
];

/** A long report keeps its opening (the lead, or its first section) and first chart in
 * view; the other sections and charts fold under "Full report". Short answers show whole. */
function split(text: string): [string, string] {
  const lines = text.split("\n");
  const headings = lines.flatMap((l, i) => (/^#{1,4}\s/.test(l) ? [i] : []));
  if (headings.length < 2) return [text, ""];
  const cut = headings[0] > 0 && lines.slice(0, headings[0]).some((l) => l.trim()) ? headings[0] : headings[1];
  return [lines.slice(0, cut).join("\n"), lines.slice(cut).join("\n")];
}

function Answer({ message }: { message: ChatMessage }) {
  const [lead, rest] = split(message.text);
  const [first, ...more] = message.charts || [];
  return (
    <>
      <Markdown text={lead} />
      {first && <Chart spec={first} width={360} />}
      {(rest || more.length > 0) && (
        <More label="Full report">
          {rest && <Markdown text={rest} />}
          {more.map((c, j) => (
            <Chart key={j} spec={c} width={360} />
          ))}
        </More>
      )}
    </>
  );
}

type Props = { chat: View["chat"]; analyst?: { model: string; thinking: string }; ask: (q: string) => Promise<boolean> };

const label = (model: string) => model.replace(/^gemini-/, "Gemini ").replace(/-flash$/, " Flash");

export default function Chat({ chat, analyst, ask }: Props) {
  const [question, setQuestion] = useState("");
  const box = useRef<HTMLDivElement>(null);
  const busy = chat.status === "running";
  useEffect(() => {
    if (box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [chat.messages.length, busy]);
  const send = async () => {
    const q = question.trim();
    if (q && (await ask(q))) setQuestion("");
  };
  return (
    <section className="card chat">
      <div className="card-head">
        <h2>Ask the analyst</h2>
        <span className="small muted" title="Set in config/config.yml (models.analyst)">
          {analyst ? `${label(analyst.model)} · ${analyst.thinking} thinking` : "stats, charts, the live model"}
        </span>
      </div>
      <div className="msgs" ref={box}>
        {chat.messages.length === 0 && (
          <div className="starters">
            <div className="muted small">Ask about the data, the features, the models or production traffic. Try:</div>
            {STARTERS.map((q) => (
              <button key={q} className="chip" disabled={busy} onClick={() => ask(q)}>
                {q}
              </button>
            ))}
          </div>
        )}
        {chat.messages.map((m, i) => (
          <div className={`msg ${m.role}`} key={i}>
            {m.role === "analyst" ? <Answer message={m} /> : (
              m.text
            )}
          </div>
        ))}
        {busy && <div className="msg analyst muted">Analysing…</div>}
      </div>
      <div className="ask">
        <input type="text" value={question} placeholder="Ask about the data…" onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => e.key === "Enter" && send()} />
        <button disabled={busy} onClick={send}>
          Ask
        </button>
      </div>
    </section>
  );
}
