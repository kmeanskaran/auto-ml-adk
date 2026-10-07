"use client";

// The four facts worth seeing first: what is live, what features exist, what past runs
// did, and when the team will ask you.

import { fmt, pct, useWords } from "@/lib/words";
import type { Config, View } from "@/lib/types";

const ASKS: Record<string, string> = { features: "features", plan: "training plan", promote: "go-live" };

export default function Overview({ view, config, open }: { view: View; config: Config; open: (tab: string) => void }) {
  const { metrics, positive } = useWords();
  const live = view.registry.production;
  const store = view.feature_store[0];
  const latest = store?.versions[0];
  const last = view.history[0];
  return (
    <div className="overview">
      <button className="tile" onClick={() => open("library")}>
        <span className="label">Live model</span>
        {live ? (
          <>
            <b>
              {live.version} <span className="muted">· {live.model}</span>
            </b>
            <span className="sub">
              catches {pct(live.test.recall)} of {positive}s · loss {fmt(metrics, "cost_per_1000", live.test.cost_per_1000)} / 1k
            </span>
          </>
        ) : (
          <>
            <b className="muted">none yet</b>
            <span className="sub">run the pipeline to put one live</span>
          </>
        )}
      </button>
      <button className="tile" onClick={() => open("features")}>
        <span className="label">Feature store</span>
        {latest ? (
          <>
            <b>
              {store.view} <span className="muted">· {latest.version}</span>
            </b>
            <span className="sub">
              {latest.features} features · {store.versions.length} version{store.versions.length > 1 ? "s" : ""}
            </span>
          </>
        ) : (
          <>
            <b className="muted">empty</b>
            <span className="sub">approved features are saved here</span>
          </>
        )}
      </button>
      <button className="tile" onClick={() => open("history")}>
        <span className="label">Past runs</span>
        {last ? (
          <>
            <b>{view.history.length} finished</b>
            <span className="sub">last: {last.outcome}</span>
          </>
        ) : (
          <>
            <b className="muted">none yet</b>
            <span className="sub">each run teaches the next</span>
          </>
        )}
      </button>
      <div className="tile static">
        <span className="label">The team asks you</span>
        <b>{config.ask_human.length ? config.ask_human.map((k) => ASKS[k] || k).join(", ") : "only when stuck"}</b>
        <span className="sub">
          after one quick review
        </span>
      </div>
    </div>
  );
}
