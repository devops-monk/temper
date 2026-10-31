import { beforeEach, describe, expect, it, vi } from "vitest";

// The store reads the page from the address when it loads; tests run without a browser.
vi.hoisted(() => Object.assign(globalThis, { location: { hash: "" } }));

import { applyOpens, routePaths, useOpened } from "./desktop";
import { useApp } from "./store";

describe("routePaths", () => {
  it("sends models to the library and data files to Datasets, and ignores the rest", () => {
    expect(routePaths(["/m/a.GGUF", "/d/qa.jsonl", "/d/t.parquet", "/n/notes.txt"])).toEqual([
      { kind: "model", path: "/m/a.GGUF" },
      { kind: "dataset", path: "/d/qa.jsonl" },
      { kind: "dataset", path: "/d/t.parquet" },
    ]);
  });
});

describe("applyOpens", () => {
  beforeEach(() => {
    useOpened.setState({ models: [], search: null, dataset: null });
    useApp.setState({ page: "chat" });
  });

  it("queues model files for Models and opens it", () => {
    applyOpens([{ kind: "model", path: "/m/a.gguf" }, { kind: "model", path: "/m/b.gguf" }]);
    expect(useOpened.getState().models).toEqual(["/m/a.gguf", "/m/b.gguf"]);
    expect(useApp.getState().page).toBe("models");
  });

  it("hands a data file to Datasets", () => {
    applyOpens([{ kind: "dataset", path: "/d/qa.csv" }]);
    expect(useOpened.getState().dataset).toBe("/d/qa.csv");
    expect(useApp.getState().page).toBe("datasets");
  });

  it("follows a link to a page, with a model search when it names one", () => {
    applyOpens([{ kind: "page", page: "train", query: null }]);
    expect(useApp.getState().page).toBe("train");
    applyOpens([{ kind: "page", page: "models", query: "Qwen/Qwen3-4B" }]);
    expect(useOpened.getState().search).toBe("Qwen/Qwen3-4B");
    expect(useApp.getState().page).toBe("models");
  });

  it("does nothing for nothing", () => {
    applyOpens([]);
    expect(useApp.getState().page).toBe("chat");
  });
});
