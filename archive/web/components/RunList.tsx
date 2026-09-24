"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { clearExperiments, type ExperimentSummary } from "@/lib/api";

export default function RunList({ initial }: { initial: ExperimentSummary[] }) {
  const router = useRouter();
  const [runs, setRuns] = useState(initial);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function clear() {
    setBusy(true);
    setError("");
    try {
      await clearExperiments();
      setRuns([]);
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not clear runs.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <aside className="card">
      <div className="card-head">
        <h2>Reviews</h2>
        <button
          type="button"
          className="btn btn-danger"
          disabled={busy || runs.length === 0}
          onClick={clear}
        >
          {busy ? "Clearing…" : "Clear runs"}
        </button>
      </div>
      <p className="lead">Open a run to watch the stages.</p>
      {error ? <div className="error">{error}</div> : null}
      <div className="list">
        {runs.length === 0 ? <div className="hint">No experiments yet.</div> : null}
        {runs.map((item) => (
          <Link className="item" key={item.id} href={`/experiments/${item.id}`}>
            <div>
              <div className="id">{item.id}</div>
              <div className="meta">{item.objective || "Classifier run"}</div>
            </div>
            <span className={`pill ${item.status}`}>{item.status.replaceAll("_", " ")}</span>
          </Link>
        ))}
      </div>
    </aside>
  );
}
