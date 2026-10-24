import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Tauri expects a fixed dev port and must not clear its own console output.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  clearScreen: false,
  server: { port: 1420, strictPort: true, watch: { ignored: ["**/src-tauri/**", "**/engine/**"] } },
  envPrefix: ["VITE_", "TAURI_ENV_"],
  build: { target: "es2022", sourcemap: !!process.env.TAURI_ENV_DEBUG },
});
