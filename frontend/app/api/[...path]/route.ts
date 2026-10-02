// Forwards /api/* to the backend's console API. BACKEND_URL is read per request, so
// the same build talks to http://backend:8000 in docker-compose and to
// http://localhost:8000 when both run on a laptop.

export const dynamic = "force-dynamic";

const backend = () => (process.env.BACKEND_URL || "http://localhost:8000").replace(/\/$/, "");

async function forward(request: Request, path: string[]): Promise<Response> {
  const url = `${backend()}/api/${path.map(encodeURIComponent).join("/")}${new URL(request.url).search}`;
  try {
    const response = await fetch(url, {
      method: request.method,
      headers: { "Content-Type": request.headers.get("Content-Type") || "application/json" },
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
