import { describe, test, expect } from "vitest";

/**
 * Tests for detectBestOllamaModel — unit tests that mock the Ollama API.
 * We test the scoring logic directly by stubbing fetch.
 */

describe("detectBestOllamaModel", () => {
  function mockFetch(models: string[]) {
    return async () => ({
      ok: true,
      json: async () => ({ models: models.map((name) => ({ name })) }),
    }) as unknown as Response;
  }

  async function detect(models: string[], baseUrl = "http://localhost:11434"): Promise<string | null> {
    const { detectBestOllamaModel } = await import("../../orchestration/ollama-detect.js");
    const origFetch = globalThis.fetch;
    (globalThis as any).fetch = mockFetch(models);
    try {
      return await detectBestOllamaModel(baseUrl);
    } finally {
      globalThis.fetch = origFetch;
    }
  }

  test("prefers qwen2.5 over llama3.2:3b", async () => {
    const result = await detect(["llama3.2:3b", "qwen2.5:7b"]);
    expect(result).toBe("qwen2.5:7b");
  });

  test("prefers qwen2.5 over llama3.1", async () => {
    const result = await detect(["llama3.1:8b", "qwen2.5:7b"]);
    expect(result).toBe("qwen2.5:7b");
  });

  test("prefers llama3.1:8b over llama3.2:3b", async () => {
    const result = await detect(["llama3.2:3b", "llama3.1:8b"]);
    expect(result).toBe("llama3.1:8b");
  });

  test("returns first model when all have equal score", async () => {
    const result = await detect(["qwen2.5:7b", "qwen2.5:14b"]);
    expect(result).toBe("qwen2.5:7b");
  });

  test("returns only available model even if low score", async () => {
    const result = await detect(["llama3.2:3b"]);
    expect(result).toBe("llama3.2:3b");
  });

  test("returns null when model list is empty", async () => {
    const result = await detect([]);
    expect(result).toBeNull();
  });

  test("returns null when Ollama is unreachable", async () => {
    const { detectBestOllamaModel } = await import("../../orchestration/ollama-detect.js");
    const origFetch = globalThis.fetch;
    (globalThis as any).fetch = async () => { throw new Error("ECONNREFUSED"); };
    try {
      const result = await detectBestOllamaModel("http://localhost:11434");
      expect(result).toBeNull();
    } finally {
      globalThis.fetch = origFetch;
    }
  });

  test("returns null on non-ok HTTP response", async () => {
    const { detectBestOllamaModel } = await import("../../orchestration/ollama-detect.js");
    const origFetch = globalThis.fetch;
    (globalThis as any).fetch = async () => ({ ok: false, status: 500 }) as unknown as Response;
    try {
      const result = await detectBestOllamaModel("http://localhost:11434");
      expect(result).toBeNull();
    } finally {
      globalThis.fetch = origFetch;
    }
  });

  test("qwen2.5-coder scores same as qwen2.5", async () => {
    const result = await detect(["llama3.1:8b", "qwen2.5-coder:7b"]);
    expect(result).toBe("qwen2.5-coder:7b");
  });
});
