import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { watchCrashes } from "./lib/crash";
import { applyTheme, useApp } from "./lib/store";
import "./index.css";

applyTheme(useApp.getState().theme);
watchCrashes();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
