// Forwards /api/* to the backend's console API. BACKEND_URL is read per request, so
// the same build talks to http://backend:8000 in docker-compose, to
// http://localhost:8000 when both run on a laptop, and to the agent on Agent Runtime
// (https://<region>-aiplatform.googleapis.com/reasoningEngines/v1/<agent>/api) when
// deployed. Google's API wants a Google access token; see googleToken().

import { execFile } from "node:child_process";
import { createHash, timingSafeEqual } from "node:crypto";
import { promisify } from "node:util";

export const dynamic = "force-dynamic";

// Deployed publicly, anyone may look (GET) but only someone with ACTION_PASSCODE may
// act (POST: start a run, answer a review, chat, put a model live, clear). Unset, as
// on a laptop, every request passes.
function mayAct(request: Request): boolean {
  const passcode = process.env.ACTION_PASSCODE;
  if (!passcode || request.method === "GET") return true;
  const digest = (text: string) => createHash("sha256").update(text).digest();
  return timingSafeEqual(digest(request.headers.get("x-passcode") || ""), digest(passcode));
}

const backend = () => (process.env.BACKEND_URL || "http://localhost:8000").replace(/\/$/, "");

const METADATA_TOKEN =
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token";
let cached: { token: string; until: number } | null = null;

// The console's own identity on Cloud Run (its service account, from the metadata
// server); on a laptop pointed at the deployed agent, your gcloud login. Kept until a
// minute before it expires: the console polls every 1.5 s.
async function googleToken(): Promise<string> {
  if (cached && Date.now() < cached.until) return cached.token;
  let token: string;
  let seconds = 3000;
  try {
    const response = await fetch(METADATA_TOKEN, {
      headers: { "Metadata-Flavor": "Google" },
      signal: AbortSignal.timeout(1000),
    });
    const body = (await response.json()) as { access_token: string; expires_in: number };
    token = body.access_token;
    seconds = body.expires_in;
  } catch {
    const { stdout } = await promisify(execFile)("gcloud", ["auth", "print-access-token"]);
    token = stdout.trim();
  }
  cached = { token, until: Date.now() + (seconds - 60) * 1000 };
  return token;
}

async function forward(request: Request, path: string[]): Promise<Response> {
  if (!mayAct(request)) {
    await new Promise((resolve) => setTimeout(resolve, 1000)); // slows down guessing
    return Response.json({ detail: "passcode" }, { status: 401 });
  }
  const base = backend();
  const url = `${base}/api/${path.map(encodeURIComponent).join("/")}${new URL(request.url).search}`;
  const headers: Record<string, string> = {
    "Content-Type": request.headers.get("Content-Type") || "application/json",
  };
  try {
    if (new URL(base).hostname.endsWith(".googleapis.com")) {
      headers.Authorization = `Bearer ${await googleToken()}`;
    }
    const response = await fetch(url, {
      method: request.method,
      headers,
      body: request.method === "GET" ? undefined : await request.text(),
      cache: "no-store",
    });
    return new Response(await response.text(), {
      status: response.status,
      headers: { "Content-Type": response.headers.get("Content-Type") || "application/json" },
    });
  } catch {
    return Response.json({ detail: `The backend is not reachable at ${backend()}.` }, { status: 502 });
  }
}

type Params = { params: Promise<{ path: string[] }> };

export async function GET(request: Request, { params }: Params) {
  return forward(request, (await params).path);
}

export async function POST(request: Request, { params }: Params) {
  return forward(request, (await params).path);
}
