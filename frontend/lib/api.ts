import type { Answer, View } from "./types";

async function call<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api/${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: "no-store",
  });
  if (!response.ok) {
    const detail = (await response.json().catch(() => ({}))).detail;
    throw new Error(typeof detail === "string" ? detail : response.statusText);
  }
  return response.json();
}

export const api = {
  view: () => call<View>("view"),
  start: (feedback = "") => call("pipeline/start", { feedback }),
  clear: () => call("clear", {}),
  pause: () => call("pipeline/pause", {}),
  resume: () => call("pipeline/resume", {}),
  restart: () => call("pipeline/restart", {}),
  setModel: (model: string, thinking: string) => call("model", { model, thinking }),
  answer: (answer: Answer) => call("pipeline/answer", answer),
  ask: (text: string) => call("chat", { text }),
  makeLive: (version: string) => call(`registry/${version}/promote`, {}),
  remove: (version: string) => call(`registry/${version}/remove`, {}),
};
