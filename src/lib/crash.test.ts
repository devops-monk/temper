import { beforeEach, describe, expect, it, vi } from "vitest";

const posted: unknown[] = [];
vi.mock("./engine", () => {
  class EngineError extends Error {}
  return {
    EngineError,
    api: vi.fn(async (_path: string, init: { json: unknown }) => {
      posted.push(init.json);
      return { id: "c1", report_url: "https://github.com/x" };
    }),
  };
});

describe("reportCrash", () => {
  beforeEach(() => {
    posted.length = 0;
    vi.stubGlobal("window", { dispatchEvent: vi.fn() });
    vi.stubGlobal("CustomEvent", class { constructor(public type: string, public init: unknown) {} });
  });

  it("saves real errors once, and leaves out the engine's answers and lost connections", async () => {
    const { reportCrash } = await import("./crash");
    const { EngineError } = await import("./engine");
    expect(await reportCrash(new RangeError("index out of range"), "chat page")).toMatchObject({ id: "c1" });
    expect(posted[0]).toMatchObject({ error: "RangeError: index out of range", context: "chat page" });
    expect(await reportCrash(new RangeError("index out of range"), "again")).toBeNull(); // once a session
    expect(await reportCrash(new (EngineError as unknown as typeof Error)("409 conflict"), "x")).toBeNull();
    expect(await reportCrash(new TypeError("Failed to fetch"), "x")).toBeNull();
    expect(await reportCrash("a string", "x")).toBeNull();
    expect(posted).toHaveLength(1);
  });
});
