"use client";

// The console: polls GET /api/view. Overview on top, the pipeline and the analyst side
// by side, and the records (activity, feature store, models, past runs) in tabs below.
// Everything it changes goes through the API.

import { useCallback, useEffect, useState } from "react";
import Activity from "./Activity";
import Chat from "./Chat";
import FeatureStore from "./FeatureStore";
import History from "./History";
import Library from "./Library";
import NewRun from "./NewRun";
import Overview from "./Overview";
import Pipeline, { WORKING } from "./Pipeline";
import { api } from "@/lib/api";
import { WordsContext, cost } from "@/lib/words";
import type { Answer, Pipeline as PipelineView, View } from "@/lib/types";

const POLL_MS = 1500;
const WAITING_ON: Record<string, string> = {
  features: "Your turn: the features",
  plan: "Your turn: the training plan",
  promote: "Your turn: go live?",
};
const TABS: [string, string][] = [
  ["activity", "Activity"],
  ["features", "Feature store"],
  ["library", "Model library"],
  ["history", "Past runs"],
];

function statusLine(p: PipelineView): [string, string] {
  const working = p.stages.find((s) => s.state === "working" || s.state === "reviewing");
  if (p.status === "running")
    return ["live", working ? (working.state === "reviewing" ? "Skeptic is reviewing…" : `${WORKING[working.key] || "Working"}…`) : "Working…"];
  if (p.status === "paused") return ["you", "Paused"];
  if (p.status === "waiting") return ["you", WAITING_ON[p.waiting_on] || "Waiting for you"];
  if (p.status === "error") return ["bad", "Run failed"];
  if (p.status === "done") return ["", "Run finished"];
  return ["", "Ready"];
}

function savedTab(): string {
  try {
    return localStorage.getItem("console-tab") || "activity";
  } catch {
    return "activity";
  }
}

export default function Console() {
  const [view, setView] = useState<View | null>(null);
  const [offline, setOffline] = useState("");
  const [tab, setTab] = useState("activity");
  const [starting, setStarting] = useState(false);

  useEffect(() => setTab(savedTab()), []);
  const open = (next: string, jump = false) => {
    setTab(next);
    try {
      localStorage.setItem("console-tab", next);
    } catch {}
    if (jump) document.getElementById("records")?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const refresh = useCallback(async () => {
    try {
      setView(await api.view());
      setOffline("");
    } catch (e) {
      setOffline(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, POLL_MS);
    return () => clearInterval(timer);
  }, [refresh]);

  const run = async (action: () => Promise<unknown>) => {
    try {
      await action();
      return true;
    } catch (e) {
      window.alert(e instanceof Error ? e.message : String(e));
      return false;
    } finally {
      refresh();
    }
  };

  if (!view) {
    return (
      <header>
        <h1>ML Team</h1>
        <span className="muted">{offline ? `Waiting for the backend: ${offline}` : "Loading…"}</span>
      </header>
    );
  }

  const p = view.pipeline;
  const cfg = p.config;
  const [dot, status] = statusLine(p);
  const words = { metrics: view.metrics, record: cfg.record || "record", positive: cfg.positive || "positive" };
  const act = async (answer: Answer) => {
    await run(() => api.answer(answer));
  };
  const busy = ["running", "paused", "waiting"].includes(p.status);
  const m = view.model;
  const restart = () => {
    if (window.confirm("Restart: stop this run wherever it is and start a new one with the same feedback?")) run(api.restart);
  };
  const clearAll = () => {
    const really = window.confirm(
      "Clear runs: a cold start. This deletes every run, model version, feature view and the analyst chat. It cannot be undone." +
        (p.status === "waiting" ? "\n\nThe review waiting for you is abandoned." : ""),
    );
    if (really) run(api.clear);
  };

  return (
    <WordsContext.Provider value={words}>
      <header>
        <div className="brand">
          <h1>ML Team</h1>
          <span className="muted">{cfg.goal || `predicts ${cfg.target}`}</span>
        </div>
        <div className="spacer" />
        <div className={`status ${dot}`}>
          <span className={`dot ${dot}`} />
          <span>{offline ? `Backend unreachable: ${offline}` : status}</span>
        </div>
        {m.available ? (
          <div className="modelpick" title="The Gemini model the team (analyst's first look, engineer, skeptic) thinks with, from its next call on. Defaults: config/config.yml">
            <span className="small muted">Team</span>
            <select aria-label="Team model" value={m.model} onChange={(e) => run(() => api.setModel(e.target.value, m.thinking))}>
              {m.models.map((o) => (
                <option key={o.key} value={o.key}>
                  {o.label}
                </option>
              ))}
            </select>
            <select aria-label="Thinking" value={m.thinking} onChange={(e) => run(() => api.setModel(m.model, e.target.value))}>
              {m.thinking_levels.map((t) => (
                <option key={t} value={t}>
                  {t} thinking
                </option>
              ))}
            </select>
          </div>
        ) : (
          <span className="small muted">{m.base}</span>
        )}
        <div className="flowctl">
          {p.status === "paused" ? (
            <button className="icon" title="Resume the run" aria-label="Resume" onClick={() => run(api.resume)}>
              ▶
            </button>
          ) : (
            <button className="icon" title="Pause the run after the step in flight" aria-label="Pause" disabled={p.status !== "running"} onClick={() => run(api.pause)}>
              ⏸
            </button>
          )}
          <button className="icon" title="Restart: a new run with the same feedback" aria-label="Restart" disabled={p.status === "idle"} onClick={restart}>
            ↻
          </button>
        </div>
        <button
          className="danger"
          disabled={p.status === "running" || view.chat.status === "running"}
          title="Cold start: delete every run, model, feature view and the chat"
          onClick={clearAll}
        >
          Clear runs
        </button>
        <button className="primary" disabled={busy} onClick={() => setStarting(true)}>
          {p.status === "idle" ? "Run pipeline" : "New run"}
        </button>
      </header>
      <NewRun open={starting} buildsOn={p.next_builds_on} onClose={() => setStarting(false)} onStart={(feedback) => run(() => api.start(feedback))} />
      <main>
        <Overview view={view} config={cfg} open={(next) => open(next, true)} />
        <div className="layout">
          <section className="card">
            <div className="card-head">
              <h2>Pipeline</h2>
              {p.run && (
                <span className="small muted" title="Time from the first step to the last (waits for you included), the model tokens the agents used, the share of them served from Gemini's prompt cache, and tool calls">
                  {p.run}
                  {p.usage && p.usage.tokens > 0 && ` · ${cost(p.usage)}`}
                </span>
              )}
            </div>
            {p.status === "error" && <div className="problems banner">{p.error}</div>}
            {p.brief && (p.brief.feedback || p.brief.builds_on) && (
              <div className="brief">
                {p.brief.builds_on ? `Rethinking ${p.brief.builds_on}'s code` : "Cold start"}
                {p.brief.feedback ? ` with your feedback: ${p.brief.feedback}` : ""}
              </div>
            )}
            <div className="small muted costline">
              Mistakes are priced: a missed {words.positive} costs <b>{cfg.costs.missed}</b>, a {cfg.false_alarm || "false alarm"} costs{" "}
              <b>{cfg.costs.false_alarm}</b>. Scores come from {words.record}s no model has seen.
            </div>
            <Pipeline pipeline={p} act={act} />
          </section>
          <aside>
            <Chat chat={view.chat} analyst={m.available ? m.analyst : undefined} ask={(q) => run(() => api.ask(q))} />
          </aside>
        </div>
        <section className="card records" id="records">
          <div className="tabs" role="tablist">
            {TABS.map(([key, label]) => (
              <button key={key} role="tab" aria-selected={tab === key} className={tab === key ? "on" : ""} onClick={() => open(key)}>
                {label}
                {key === "library" && view.registry.versions.length > 0 && <span className="count">{view.registry.versions.length}</span>}
                {key === "history" && view.history.length > 0 && <span className="count">{view.history.length}</span>}
              </button>
            ))}
          </div>
          {tab === "activity" && <Activity lines={p.activity} run={p.run} usage={p.usage} />}
          {tab === "features" && <FeatureStore views={view.feature_store} />}
          {tab === "library" && <Library versions={view.registry.versions} onLive={(v) => run(() => api.makeLive(v))} onRemove={(v) => run(() => api.remove(v))} />}
          {tab === "history" && <History runs={view.history} />}
        </section>
      </main>
    </WordsContext.Provider>
  );
}
