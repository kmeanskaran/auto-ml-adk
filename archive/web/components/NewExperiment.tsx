"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { startRun } from "@/lib/api";

export default function NewExperiment() {
  const router = useRouter();
  const [objective, setObjective] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!file) {
      setError("Upload a CSV.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const board = await startRun({ file, objective });
      router.push(`/experiments/${board.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not start the run.");
      setBusy(false);
    }
  }

  return (
    <form className="card" onSubmit={onSubmit}>
      <h2>The question</h2>
      <p className="lead">
        Upload the table and describe the outcome. The column to predict, and which scores
        apply, are decided after the table is read.
      </p>
      <label>Table</label>
      <label className="drop">
        <strong>{file ? file.name : "Drop a CSV or click to browse"}</strong>
        The first row should be the column names.
        <input
          type="file"
          accept=".csv,text/csv"
          onChange={(event) => setFile(event.target.files?.[0] || null)}
        />
      </label>
      <label>Question</label>
      <textarea
        value={objective}
        onChange={(event) => setObjective(event.target.value)}
        placeholder="What should this table predict?"
        required
      />
      <div className="form-actions">
        <button className="btn btn-primary" type="submit" disabled={busy}>
          {busy ? "Starting…" : "Start"}
        </button>
      </div>
      {error ? <div className="error">{error}</div> : null}
    </form>
  );
}
