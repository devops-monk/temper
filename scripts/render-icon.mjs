// Renders assets/icon.svg to a 1024px PNG; `tauri icon` then derives every platform size.
// `--clean` removes the Android/iOS sets `tauri icon` also writes, which a desktop app does not ship.
import { readFileSync, writeFileSync, copyFileSync, rmSync } from "node:fs";
import { Resvg } from "@resvg/resvg-js";

if (process.argv.includes("--clean")) {
  for (const dir of ["android", "ios"]) rmSync(`src-tauri/icons/${dir}`, { recursive: true, force: true });
} else {
  const png = new Resvg(readFileSync("assets/icon.svg"), { fitTo: { mode: "width", value: 1024 } }).render().asPng();
  writeFileSync("assets/icon-1024.png", png);
  copyFileSync("assets/icon.svg", "public/icon.svg");
  console.log("wrote assets/icon-1024.png");
}
