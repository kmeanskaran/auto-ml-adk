"use client";

// Starting a run: optional feedback for the team. A run builds on the last finished
// run's code, which the team rethinks with the feedback; with no runs left (after
// "Clear runs") it is a cold start.

import { useEffect, useRef, useState } from "react";

type Props = {
  open: boolean;
  buildsOn: string | null;
  onClose: () => void;
  onStart: (feedback: string) => Promise<boolean>;
};

export default function NewRun({ open, buildsOn, onClose, onStart }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [feedback, setFeedback] = useState("");
  const [sending, setSending] = useState(false);

  useEffect(() => {
    const d = dialog.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);

  const start = async () => {
    setSending(true);
    const ok = await onStart(feedback.trim());
    setSending(false);
    if (ok) {
      setFeedback("");
      onClose();
    }
  };

  return (
    <dialog ref={dialog} className="newrun" onClose={onClose}>
      <h2>New run</h2>
      <p className="small muted">
        {buildsOn ? (
          <>
            Builds on <b>{buildsOn.replace(/^run-/, "")}</b>: the team reads that run&apos;s code and rethinks it with your feedback.
          </>
        ) : (
          "Cold start: no earlier run to build on, so the team starts from the data."
        )}
      </p>
      <label className="small" htmlFor="run-feedback">
        Feedback for the team <span className="muted">(optional)</span>
      </label>
      <textarea
        id="run-feedback"
        rows={4}
        value={feedback}
        maxLength={4000}
        autoFocus
        placeholder="What should the team rethink or try this time?"
        onChange={(e) => setFeedback(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && !sending) start();
        }}
      />
      <div className="actions">
        <button onClick={onClose} disabled={sending}>
          Cancel
        </button>
        <button className="primary" onClick={start} disabled={sending}>
          {sending ? "Starting…" : feedback.trim() ? "Start with feedback" : "Start run"}
        </button>
      </div>
    </dialog>
  );
}
