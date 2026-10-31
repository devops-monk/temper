import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Tauri expects a fixed dev port and must not clear its own console output.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  clearScreen: false,
  server: { port: 1420, strictPort: true, watch: { ignored: ["**/src-tauri/**", "**/engine/**"] } },
  envPrefix: ["VITE_", "TAURI_ENV_"],
  // Relative asset paths, so the same build works in the app and at the engine's /ui/ (`temper serve`).
  base: "./",
  build: { target: "es2022", sourcemap: !!process.env.TAURI_ENV_DEBUG },
});
