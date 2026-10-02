import type { NextConfig } from "next";

// "standalone" builds a self-contained server for the Docker image (see Dockerfile).
// The backend is reached through app/api/[...path]/route.ts, which reads BACKEND_URL
// at runtime, so one image works in docker-compose and on a laptop.
const config: NextConfig = {
  output: "standalone",
  reactStrictMode: true,
};

export default config;
