import NewExperiment from "@/components/NewExperiment";
import RunList from "@/components/RunList";
import { listExperiments } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  let experiments: Awaited<ReturnType<typeof listExperiments>> = [];
  let listError = "";
  try {
    experiments = await listExperiments();
  } catch (error) {
    listError = error instanceof Error ? error.message : "API is offline.";
  }
  return (
    <main className="shell">
      <header className="mast">
        <p className="eyebrow">New run</p>
        <h1>What should the model answer?</h1>
        <p className="hint">
          Upload a table and write the question. Data engineering loads the scores on the
          next stage, after it has seen the data.
        </p>
      </header>
      <div className="grid">
        <NewExperiment />
        {listError ? (
          <aside className="card">
            <h2>Reviews</h2>
            <div className="error">{listError}</div>
          </aside>
        ) : (
          <RunList initial={experiments} />
        )}
      </div>
    </main>
  );
}
