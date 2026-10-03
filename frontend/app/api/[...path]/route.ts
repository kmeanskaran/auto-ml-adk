// Forwards /api/* to the backend's console API. BACKEND_URL is read per request, so
// the same build talks to http://backend:8000 in docker-compose, to
// http://localhost:8000 when both run on a laptop, and to the agent on Agent Runtime
// (https://<region>-aiplatform.googleapis.com/reasoningEngines/v1/<agent>/api) when
// deployed. Google's API wants a Google access token; see googleToken().

import { execFile } from "node:child_process";
import { promisify } from "node:util";

export const dynamic = "force-dynamic";

const backend = () => (process.env.BACKEND_URL || "http://localhost:8000").replace(/\/$/, "");
const onGoogle = (base: string) => new URL(base).hostname.endsWith(".googleapis.com");

const METADATA_TOKEN =
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token";
let cached: { token: string; until: number } | null = null;

// The console's own identity on Cloud Run (its service account, from the metadata
// server); on a laptop pointed at the deployed agent, your gcloud login. Kept until a
// minute before it expires.
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

type Reply = { status: number; body: string; type: string };

async function send(request: Request, url: string, base: string): Promise<Reply> {
  const headers: Record<string, string> = {
    "Content-Type": request.headers.get("Content-Type") || "application/json",
  };
  if (onGoogle(base)) headers.Authorization = `Bearer ${await googleToken()}`;
  const response = await fetch(url, {
    method: request.method,
    headers,
    body: request.method === "GET" ? undefined : await request.text(),
    cache: "no-store",
  });
  return {
    status: response.status,
    body: await response.text(),
    type: response.headers.get("Content-Type") || "application/json",
  };
}

// Agent Runtime lets one caller through about 30 times a minute, and every open tab
// polls every 1.5 s. Deployed, the console asks the agent for a view at most every
// VIEW_SECONDS and shares the answer across tabs; when Google still says 429, the
// tabs keep the last good view instead of an error.
const VIEW_SECONDS = 3;
const views = new Map<string, { reply: Reply; at: number }>();
const pending = new Map<string, Promise<Reply>>();

async function view(request: Request, url: string, base: string): Promise<Reply> {
  const last = views.get(url);
  if (last && Date.now() - last.at < VIEW_SECONDS * 1000) return last.reply;
  let asking = pending.get(url);
  if (!asking) {
    asking = send(request, url, base).finally(() => pending.delete(url));
    pending.set(url, asking);
  }
  const reply = await asking;
  if (reply.status === 200) views.set(url, { reply, at: Date.now() });
  else if (reply.status === 429 && last) return last.reply;
  return reply;
}

async function forward(request: Request, path: string[]): Promise<Response> {
  const base = backend();
  const url = `${base}/api/${path.map(encodeURIComponent).join("/")}${new URL(request.url).search}`;
  try {
    const reply =
      request.method === "GET" && onGoogle(base)
        ? await view(request, url, base)
        : await send(request, url, base);
    return new Response(reply.body, { status: reply.status, headers: { "Content-Type": reply.type } });
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
