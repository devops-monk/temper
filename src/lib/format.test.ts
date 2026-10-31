import { describe, expect, it } from "vitest";
import { bytes, duration, ollamaName, quantLabel } from "./format";

describe("bytes", () => {
  it("picks a unit and keeps one decimal below 100", () => {
    expect(bytes(0)).toBe("0 B");
    expect(bytes(512)).toBe("512 B");
    expect(bytes(1536)).toBe("1.5 KB");
    expect(bytes(2.5 * 1024 ** 3)).toBe("2.5 GB");
    expect(bytes(150 * 1024 ** 2)).toBe("150 MB");
  });
});

describe("duration", () => {
  it("reads as seconds, minutes or hours", () => {
    expect(duration(null)).toBe("–");
    expect(duration(42.4)).toBe("42s");
    expect(duration(125)).toBe("2m 5s");
    expect(duration(3 * 3600 + 7 * 60)).toBe("3h 7m");
  });
});

describe("quantLabel", () => {
  it("finds the quant in a GGUF's file name", () => {
    expect(quantLabel("Qwen3-8B-UD-Q4_K_XL.gguf")).toBe("UD-Q4_K_XL");
    expect(quantLabel("models/run-imatrix-IQ3_M.gguf")).toBe("IQ3_M");
    expect(quantLabel("big-Q8_0-00001-of-00003.gguf")).toBe("Q8_0");
    expect(quantLabel("tiny-F16.gguf")).toBe("F16");
    expect(quantLabel("no-quant-here.gguf")).toBe("no-quant-here");
  });
});

describe("ollamaName", () => {
  it("names a fine-tune's GGUF the way Ollama expects", () => {
    expect(ollamaName("/exports/x/gguf/20261024-154903-gsm8k-math-tutor-Q4_K_M.gguf")).toBe("gsm8k-math-tutor:q4_k_m");
    expect(ollamaName("C:\\models\\My Model-IQ3_M.gguf")).toBe("my-model:iq3_m");
    expect(ollamaName("plain.gguf")).toBe("plain");
    expect(ollamaName("---.gguf")).toBe("model");
  });
});
