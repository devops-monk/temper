import { defineConfig } from "@playwright/test";

// End-to-end tests: the real interface in Chrome against a real engine with an empty home folder, so
// they start from a first launch. `npm run test:e2e` starts both; nothing needs to be running.
const ENGINE = 18765;
const UI = 14210;
const HOME = "e2e/.home";

export default defineConfig({
  testDir: "e2e",
  timeout: 30_000,
  retries: process.env.CI ? 1 : 0,
  workers: 1, // one engine, shared state
  use: { baseURL: `http://localhost:${UI}`, channel: "chrome", viewport: { width: 1320, height: 840 }, trace: "retain-on-failure" },
  webServer: [
    {
      // Each run starts from an empty home folder, like a first launch.
      command: `node -e "require('fs').rmSync('${HOME}', { recursive: true, force: true })" && uv --directory engine run temper-engine --port ${ENGINE} --token e2e`,
      url: `http://127.0.0.1:${ENGINE}/api/health`,
      env: { TEMPER_HOME: `${process.cwd()}/${HOME}`, TEMPER_ORIGINS: `http://localhost:${UI}` },
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: `npx vite --port ${UI} --strictPort`,
      url: `http://localhost:${UI}`,
      env: { VITE_ENGINE_URL: `http://127.0.0.1:${ENGINE}`, VITE_ENGINE_TOKEN: "e2e" },
      reuseExistingServer: false,
      timeout: 60_000,
    },
  ],
});
