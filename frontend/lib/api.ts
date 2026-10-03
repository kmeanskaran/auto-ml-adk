import type { Answer, View } from "./types";

// A public deployment asks for a passcode before any action (see the /api proxy). It is
// asked once, kept for this browser tab, and asked again if it was wrong.
const PASSCODE = "console-passcode";

function passcode(): string {
  try {
    return sessionStorage.getItem(PASSCODE) || "";
  } catch {
    return "";
  }
}

async function call<T>(path: string, body?: unknown, retry = true): Promise<T> {
  const response = await fetch(`/api/${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json", "x-passcode": passcode() },
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: "no-store",
  });
  if (response.status === 401 && retry) {
    const typed = window.prompt(passcode() ? "Wrong passcode. Try again:" : "Passcode to act on this console:");
    if (!typed) throw new Error("Viewing only: actions need the passcode.");
    try {
      sessionStorage.setItem(PASSCODE, typed);
    } catch {}
    return call<T>(path, body, true);
  }
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
