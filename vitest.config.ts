import { defineConfig } from "vitest/config";

// Unit tests for the interface's own logic; the end-to-end tests in e2e/ run with Playwright.
export default defineConfig({
  test: { include: ["src/**/*.test.ts"], environment: "node" },
});
