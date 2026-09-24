import Link from "next/link";
import StageBoard from "@/components/StageBoard";
import { getBoard } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function ExperimentPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const board = await getBoard(id);
  return (
    <main className="shell">
      <header className="topbar">
        <Link className="hint" href="/">
          ← New run
        </Link>
      </header>
      <StageBoard initial={board} />
    </main>
  );
}
