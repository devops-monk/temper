import { useEffect } from "react";
import { Rail } from "./components/Rail";
import { StatusBar } from "./components/StatusBar";
import { EngineGate } from "./components/EngineGate";
import { useApp } from "./lib/store";
import { ChatPage } from "./pages/Chat";
import { ConnectPage } from "./pages/Connect";
import { DatasetsPage } from "./pages/Datasets";
import { ExportPage } from "./pages/Export";
import { ModelsPage } from "./pages/Models";
import { RecipesPage } from "./pages/Recipes";
import { SettingsPage } from "./pages/Settings";
import { TrainPage } from "./pages/Train";

/** Poll while the window is visible; skip ticks when it is hidden. */
function usePoll(fn: () => void, ms: number) {
  useEffect(() => {
    fn();
    const id = setInterval(() => document.visibilityState === "visible" && fn(), ms);
    return () => clearInterval(id);
  }, [fn, ms]);
}

export function App() {
  const { page, engine, refreshEngine, refreshSystem, refreshRuntime, refreshJobs } = useApp();
  const ready = engine?.status === "running";

  usePoll(refreshEngine, ready ? 10_000 : 1_000);
  useEffect(() => {
    if (!ready) return;
    const tick = () => {
      if (document.visibilityState !== "visible") return;
      void refreshSystem();
      void refreshRuntime();
      void refreshJobs();
    };
    const id = setInterval(tick, 4_000);
    return () => clearInterval(id);
  }, [ready, refreshSystem, refreshRuntime, refreshJobs]);

  return (
    <div className="flex h-full flex-col">
      <div className="flex min-h-0 flex-1">
        <Rail />
        <main className="flex min-w-0 flex-1 flex-col bg-bg">
          {page === "settings" ? (
            <SettingsPage />
          ) : !ready ? (
            <EngineGate />
          ) : (
            <>
              {page === "chat" && <ChatPage />}
              {page === "models" && <ModelsPage />}
              {page === "train" && <TrainPage />}
              {page === "datasets" && <DatasetsPage />}
              {page === "recipes" && <RecipesPage />}
              {page === "export" && <ExportPage />}
              {page === "connect" && <ConnectPage />}
            </>
          )}
        </main>
      </div>
      <StatusBar />
    </div>
  );
}
