import { EventEmitter } from "node:events";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type BetterSqlite3 from "better-sqlite3";
import type { TaskNode } from "../../orchestration/task-graph.js";

// ─── Mocks ──────────────────────────────────────────────────────────────────

vi.mock("node:child_process", () => ({
  spawn: vi.fn(),
  exec: vi.fn((_cmd: string, cb: (err: Error | null, stdout: string, stderr: string) => void) => {
    cb(null, "/usr/local/bin/goose", "");
  }),
}));

vi.mock("../../orchestration/task-graph.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../orchestration/task-graph.js")>();
  return {
    ...actual,
    completeTask: vi.fn(),
    failTask: vi.fn(),
  };
});

// ─── Helpers ────────────────────────────────────────────────────────────────

/**
 * Create a minimal fake ChildProcess whose events fire after a microtask tick.
 * `exitCode` controls what `exit` emits; `errorToEmit` causes an `error` event instead.
 */
function makeProc({
  exitCode = 0,
  stdout = "",
  stderr = "",
  errorToEmit,
  delayMs = 0,
}: {
  exitCode?: number;
  stdout?: string;
  stderr?: string;
  errorToEmit?: Error;
  delayMs?: number;
} = {}) {
  const proc = new EventEmitter() as any;
  proc.stdout = new EventEmitter();
  proc.stderr = new EventEmitter();
  proc.killed = false;
  proc.kill = vi.fn((_sig?: string) => { proc.killed = true; });

  const fire = () => {
    if (errorToEmit) {
      proc.emit("error", errorToEmit);
    } else {
      if (stdout) proc.stdout.emit("data", Buffer.from(stdout));
      if (stderr) proc.stderr.emit("data", Buffer.from(stderr));
      proc.emit("exit", exitCode);
    }
  };

  if (delayMs > 0) {
    setTimeout(fire, delayMs);
  } else {
    setImmediate(fire);
  }

  return proc;
}

function makeTask(overrides: Partial<TaskNode> = {}): TaskNode {
  return {
    id: "task-01",
    parentId: null,
    goalId: "goal-01",
    title: "Write hello world",
    description: "Write a hello world script",
    status: "assigned",
    assignedTo: "goose-worker-abc",
    agentRole: "generalist",
    priority: 50,
    dependencies: [],
    result: null,
    metadata: {
      estimatedCostCents: 0,
      actualCostCents: 0,
      maxRetries: 1,
      retryCount: 0,
      timeoutMs: 5_000,
      createdAt: new Date().toISOString(),
      startedAt: null,
      completedAt: null,
    },
    ...overrides,
  };
}

// Minimal DB stub — GooseWorkerPool only forwards it to completeTask/failTask,
// both of which are mocked, so we never need real SQLite here.
function makeDb(): BetterSqlite3.Database {
  return {} as unknown as BetterSqlite3.Database;
}

// ─── Tests ──────────────────────────────────────────────────────────────────

describe("GooseWorkerPool", () => {
  let db: BetterSqlite3.Database;
  let spawnMock: ReturnType<typeof vi.fn>;
  let completeTask: ReturnType<typeof vi.fn>;
  let failTask: ReturnType<typeof vi.fn>;

  beforeEach(async () => {
    db = makeDb();
    vi.clearAllMocks();

    const childProcess = await import("node:child_process");
    spawnMock = childProcess.spawn as unknown as ReturnType<typeof vi.fn>;

    const taskGraph = await import("../../orchestration/task-graph.js");
    completeTask = taskGraph.completeTask as unknown as ReturnType<typeof vi.fn>;
    failTask = taskGraph.failTask as unknown as ReturnType<typeof vi.fn>;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // ── spawn() shape ──────────────────────────────────────────────────────────

  describe("spawn()", () => {
    it("returns correct address/name/sandboxId shape", async () => {
      spawnMock.mockReturnValue(makeProc());
      const { GooseWorkerPool } = await import("../../orchestration/goose-worker.js");

      const pool = new GooseWorkerPool({ db, provider: "ollama", model: "qwen3.5:4b", workerId: "pool-test" });
      const result = pool.spawn(makeTask());

      expect(result.address).toMatch(/^local:\/\/goose-worker-/);
      expect(result.name).toMatch(/^goose-generalist-/);
      expect(result.sandboxId).toMatch(/^goose-worker-/);
      expect(result.sandboxId).toBe(result.address.replace("local://", ""));

      await pool.shutdown();
    });

    it("passes provider and model to goose CLI args", async () => {
      spawnMock.mockReturnValue(makeProc());
      const { GooseWorkerPool } = await import("../../orchestration/goose-worker.js");

      const pool = new GooseWorkerPool({ db, provider: "anthropic", model: "claude-sonnet-4-6", workerId: "pool-test" });
      pool.spawn(makeTask());
      await new Promise((r) => setImmediate(r)); // let the process fire

      expect(spawnMock).toHaveBeenCalledWith(
        "goose",
        expect.arrayContaining(["--provider", "anthropic", "--model", "claude-sonnet-4-6", "--output-format", "json"]),
        expect.any(Object),
      );

      await pool.shutdown();
    });

    it("includes task title and description in the -t prompt", async () => {
      spawnMock.mockReturnValue(makeProc());
      const { GooseWorkerPool } = await import("../../orchestration/goose-worker.js");

      const task = makeTask({ title: "Build API", description: "Create REST endpoints" });
      const pool = new GooseWorkerPool({ db, provider: "ollama", model: "qwen3.5:4b", workerId: "pool-test" });
      pool.spawn(task);
      await new Promise((r) => setImmediate(r));

      const args: string[] = spawnMock.mock.calls[0][1];
      const tIndex = args.indexOf("-t");
      const prompt = args[tIndex + 1];
      expect(prompt).toContain("Build API");
      expect(prompt).toContain("Create REST endpoints");

      await pool.shutdown();
    });
  });

  // ── hasWorker() / getActiveCount() ────────────────────────────────────────

  describe("hasWorker() / getActiveCount()", () => {
    it("hasWorker() returns true immediately after spawn, before process exits", async () => {
      const proc = makeProc({ delayMs: 50 }); // slow process
      spawnMock.mockReturnValue(proc);
      const { GooseWorkerPool } = await import("../../orchestration/goose-worker.js");

      const pool = new GooseWorkerPool({ db, provider: "ollama", model: "qwen3.5:4b", workerId: "pool-test" });
      const { address } = pool.spawn(makeTask());

      // Synchronously after spawn — worker must be visible
      expect(pool.hasWorker(address)).toBe(true);
      expect(pool.getActiveCount()).toBe(1);

      await pool.shutdown();
    });

    it("hasWorker() accepts raw workerId without the local:// prefix", async () => {
      spawnMock.mockReturnValue(makeProc({ delayMs: 50 }));
      const { GooseWorkerPool } = await import("../../orchestration/goose-worker.js");

      const pool = new GooseWorkerPool({ db, provider: "ollama", model: "qwen3.5:4b", workerId: "pool-test" });
      const { address } = pool.spawn(makeTask());
      const rawId = address.replace("local://", "");

      expect(pool.hasWorker(rawId)).toBe(true);

      await pool.shutdown();
    });

    it("hasWorker() returns false after process exits", async () => {
      spawnMock.mockReturnValue(makeProc({ exitCode: 0, stdout: "done" }));
      const { GooseWorkerPool } = await import("../../orchestration/goose-worker.js");

      const pool = new GooseWorkerPool({ db, provider: "ollama", model: "qwen3.5:4b", workerId: "pool-test" });
      const { address } = pool.spawn(makeTask());

      // Drain the microtask queue enough times for: setImmediate → exit handler
      // → completeTask → resolve → finally → activeWorkers.delete
      for (let i = 0; i < 20; i++) {
        await new Promise((r) => setImmediate(r));
      }

      expect(pool.hasWorker(address)).toBe(false);
      expect(pool.getActiveCount()).toBe(0);
    });

    it("tracks multiple concurrent workers", async () => {
      spawnMock.mockReturnValue(makeProc({ delayMs: 50 }));
      const { GooseWorkerPool } = await import("../../orchestration/goose-worker.js");

      const pool = new GooseWorkerPool({ db, provider: "ollama", model: "qwen3.5:4b", workerId: "pool-test", maxWorkers: 3 });
      pool.spawn(makeTask({ id: "t1" }));
      pool.spawn(makeTask({ id: "t2" }));
      pool.spawn(makeTask({ id: "t3" }));

      expect(pool.getActiveCount()).toBe(3);

      await pool.shutdown();
    });
  });

  // ── Success path ──────────────────────────────────────────────────────────

  describe("successful completion", () => {
    it("calls completeTask with success=true on exit code 0", async () => {
      spawnMock.mockReturnValue(makeProc({ exitCode: 0, stdout: "Task done!" }));
      const { GooseWorkerPool } = await import("../../orchestration/goose-worker.js");

      const pool = new GooseWorkerPool({ db, provider: "ollama", model: "qwen3.5:4b", workerId: "pool-test" });
      const task = makeTask();
      pool.spawn(task);

      await new Promise((r) => setTimeout(r, 20));

      expect(completeTask).toHaveBeenCalledOnce();
      const [, taskId, result] = completeTask.mock.calls[0];
      expect(taskId).toBe(task.id);
      expect(result.success).toBe(true);
      expect(result.output).toBe("Task done!");
      expect(result.costCents).toBe(0);
      expect(typeof result.duration).toBe("number");
    });

    it("parses last assistant message from primary { messages: [...] } format", async () => {
      const jsonOutput = JSON.stringify({
        messages: [
          { role: "user", content: "Do the task" },
          { role: "assistant", content: "First attempt." },
          { role: "assistant", content: "Final result: done." },
        ],
        metadata: { total_tokens: 100, status: "completed" },
      });
      spawnMock.mockReturnValue(makeProc({ exitCode: 0, stdout: jsonOutput }));
      const { GooseWorkerPool } = await import("../../orchestration/goose-worker.js");

      const pool = new GooseWorkerPool({ db, provider: "ollama", model: "qwen3.5:4b", workerId: "pool-test" });
      pool.spawn(makeTask());

      await new Promise((r) => setTimeout(r, 20));

      expect(completeTask).toHaveBeenCalledOnce();
      const result = completeTask.mock.calls[0][2];
      expect(result.output).toBe("Final result: done.");
    });

    it("parses content blocks [{ type: 'text', text: '...' }] in assistant messages", async () => {
      const jsonOutput = JSON.stringify({
        messages: [
          { role: "assistant", content: [{ type: "text", text: "Part A." }, { type: "text", text: "Part B." }] },
        ],
      });
      spawnMock.mockReturnValue(makeProc({ exitCode: 0, stdout: jsonOutput }));
      const { GooseWorkerPool } = await import("../../orchestration/goose-worker.js");

      const pool = new GooseWorkerPool({ db, provider: "ollama", model: "qwen3.5:4b", workerId: "pool-test" });
      pool.spawn(makeTask());

      await new Promise((r) => setTimeout(r, 20));

      const result = completeTask.mock.calls[0][2];
      expect(result.output).toBe("Part A.\nPart B.");
    });

    it("falls back to bare array of messages", async () => {
      const jsonOutput = JSON.stringify([
        { role: "user", content: "Do the task" },
        { role: "assistant", content: "Bare array result." },
      ]);
      spawnMock.mockReturnValue(makeProc({ exitCode: 0, stdout: jsonOutput }));
      const { GooseWorkerPool } = await import("../../orchestration/goose-worker.js");

      const pool = new GooseWorkerPool({ db, provider: "ollama", model: "qwen3.5:4b", workerId: "pool-test" });
      pool.spawn(makeTask());

      await new Promise((r) => setTimeout(r, 20));

      const result = completeTask.mock.calls[0][2];
      expect(result.output).toBe("Bare array result.");
    });

    it("falls back to raw stdout when output is plain text (older Goose)", async () => {
      spawnMock.mockReturnValue(makeProc({ exitCode: 0, stdout: "plain text result" }));
      const { GooseWorkerPool } = await import("../../orchestration/goose-worker.js");

      const pool = new GooseWorkerPool({ db, provider: "ollama", model: "qwen3.5:4b", workerId: "pool-test" });
      pool.spawn(makeTask());

      await new Promise((r) => setTimeout(r, 20));

      const result = completeTask.mock.calls[0][2];
      expect(result.output).toBe("plain text result");
    });

    it("returns a default message when stdout is empty", async () => {
      spawnMock.mockReturnValue(makeProc({ exitCode: 0, stdout: "" }));
      const { GooseWorkerPool } = await import("../../orchestration/goose-worker.js");

      const pool = new GooseWorkerPool({ db, provider: "ollama", model: "qwen3.5:4b", workerId: "pool-test" });
      pool.spawn(makeTask());

      await new Promise((r) => setTimeout(r, 20));

      const result = completeTask.mock.calls[0][2];
      expect(result.output).toBe("Task completed (no output).");
    });
  });

  // ── Failure path ──────────────────────────────────────────────────────────

  describe("failure handling", () => {
    it("calls failTask on non-zero exit code", async () => {
      spawnMock.mockReturnValue(makeProc({ exitCode: 1, stderr: "Something went wrong" }));
      const { GooseWorkerPool } = await import("../../orchestration/goose-worker.js");

      const pool = new GooseWorkerPool({ db, provider: "ollama", model: "qwen3.5:4b", workerId: "pool-test" });
      const task = makeTask();
      pool.spawn(task);

      await new Promise((r) => setTimeout(r, 20));

      expect(failTask).toHaveBeenCalledOnce();
      const [, taskId, errorMsg, retryable] = failTask.mock.calls[0];
      expect(taskId).toBe(task.id);
      expect(errorMsg).toContain("code 1");
      expect(errorMsg).toContain("Something went wrong");
      expect(retryable).toBe(true);
    });

    it("calls failTask when goose binary is not found (ENOENT)", async () => {
      const enoent = Object.assign(new Error("spawn goose ENOENT"), { code: "ENOENT", message: "ENOENT" });
      spawnMock.mockReturnValue(makeProc({ errorToEmit: enoent }));
      const { GooseWorkerPool } = await import("../../orchestration/goose-worker.js");

      const pool = new GooseWorkerPool({ db, provider: "ollama", model: "qwen3.5:4b", workerId: "pool-test" });
      const task = makeTask();
      pool.spawn(task);

      await new Promise((r) => setTimeout(r, 20));

      expect(failTask).toHaveBeenCalledOnce();
      const [, taskId, , retryable] = failTask.mock.calls[0];
      expect(taskId).toBe(task.id);
      expect(retryable).toBe(true);
    });

    it("calls failTask on process error event", async () => {
      const err = new Error("process crashed unexpectedly");
      spawnMock.mockReturnValue(makeProc({ errorToEmit: err }));
      const { GooseWorkerPool } = await import("../../orchestration/goose-worker.js");

      const pool = new GooseWorkerPool({ db, provider: "ollama", model: "qwen3.5:4b", workerId: "pool-test" });
      const task = makeTask();
      pool.spawn(task);

      await new Promise((r) => setTimeout(r, 20));

      expect(failTask).toHaveBeenCalledOnce();
      const [, taskId, errorMsg] = failTask.mock.calls[0];
      expect(taskId).toBe(task.id);
      expect(errorMsg).toContain("process crashed unexpectedly");
    });

    it("does not call completeTask on non-zero exit", async () => {
      spawnMock.mockReturnValue(makeProc({ exitCode: 2 }));
      const { GooseWorkerPool } = await import("../../orchestration/goose-worker.js");

      const pool = new GooseWorkerPool({ db, provider: "ollama", model: "qwen3.5:4b", workerId: "pool-test" });
      pool.spawn(makeTask());

      await new Promise((r) => setTimeout(r, 20));

      expect(completeTask).not.toHaveBeenCalled();
    });
  });

  // ── Timeout ───────────────────────────────────────────────────────────────

  describe("timeout", () => {
    it("kills the process and calls failTask when timeoutMs elapses", async () => {
      // Process that never emits exit on its own
      const proc = new EventEmitter() as any;
      proc.stdout = new EventEmitter();
      proc.stderr = new EventEmitter();
      proc.killed = false;
      proc.kill = vi.fn((_sig?: string) => { proc.killed = true; });
      spawnMock.mockReturnValue(proc);

      const { GooseWorkerPool } = await import("../../orchestration/goose-worker.js");

      const pool = new GooseWorkerPool({ db, provider: "ollama", model: "qwen3.5:4b", workerId: "pool-test" });
      // Very short timeout so the real timer fires quickly
      const task = makeTask({ metadata: { timeoutMs: 50 } as any });
      pool.spawn(task);

      // Wait past the timeout
      await new Promise((r) => setTimeout(r, 150));

      expect(proc.kill).toHaveBeenCalled();
      expect(failTask).toHaveBeenCalledOnce();
      const [, taskId, errorMsg] = failTask.mock.calls[0];
      expect(taskId).toBe(task.id);
      expect(errorMsg).toContain("timed out");

      await pool.shutdown();
    });
  });

  // ── shutdown() ────────────────────────────────────────────────────────────

  describe("hollow-completion detection", () => {
    function makeGooseJson(messages: unknown[]): string {
      const banner = "__( O)>\n\\____)\nL L\n";
      return banner + JSON.stringify({ messages, metadata: { status: "completed" } });
    }

    function makeToolResponse(): unknown {
      return {
        role: "user",
        content: [{ type: "toolResponse", id: "call_1", toolResult: { status: "success", value: {} } }],
      };
    }

    function makeAssistantText(text: string): unknown {
      return { role: "assistant", content: [{ type: "text", text }] };
    }

    it("marks task complete when tools were called (not hollow)", async () => {
      const messages = [
        makeAssistantText("I'll write the file now."),
        makeToolResponse(),
        makeAssistantText("Done! I wrote the file successfully."),
      ];
      const stdout = makeGooseJson(messages);
      spawnMock.mockReturnValueOnce(makeProc({ stdout }));
      const { GooseWorkerPool } = await import("../../orchestration/goose-worker.js");
      const pool = new GooseWorkerPool({ db, provider: "ollama", model: "qwen2.5:7b", workerId: "pool-test" });

      pool.spawn(makeTask());
      await new Promise((r) => setTimeout(r, 50));

      expect(completeTask).toHaveBeenCalled();
      expect(failTask).not.toHaveBeenCalled();
    });

    it("fails task when no tools were called and output looks hollow", async () => {
      const hollowText = "I would write this file. You should follow these steps:\n```bash\necho hello > file.txt\n```";
      const messages = [makeAssistantText(hollowText)];
      const stdout = makeGooseJson(messages);
      spawnMock.mockReturnValueOnce(makeProc({ stdout }));
      const { GooseWorkerPool } = await import("../../orchestration/goose-worker.js");
      const pool = new GooseWorkerPool({ db, provider: "ollama", model: "llama3.2:3b", workerId: "pool-test" });

      pool.spawn(makeTask());
      await new Promise((r) => setTimeout(r, 50));

      expect(failTask).toHaveBeenCalledWith(
        db,
        "task-01",
        expect.stringContaining("described the task without executing it"),
        true,
      );
      expect(completeTask).not.toHaveBeenCalled();
    });

    it("marks complete when no tools called but output has no hollow phrases", async () => {
      // Model gave a genuine text summary with no code blocks → not hollow
      const messages = [makeAssistantText("The task is complete. All configuration was verified.")];
      const stdout = makeGooseJson(messages);
      spawnMock.mockReturnValueOnce(makeProc({ stdout }));
      const { GooseWorkerPool } = await import("../../orchestration/goose-worker.js");
      const pool = new GooseWorkerPool({ db, provider: "ollama", model: "qwen2.5:7b", workerId: "pool-test" });

      pool.spawn(makeTask());
      await new Promise((r) => setTimeout(r, 50));

      expect(completeTask).toHaveBeenCalled();
      expect(failTask).not.toHaveBeenCalled();
    });
  });

  describe("shutdown()", () => {
    it("kills all active workers and clears the map", async () => {
      const proc1 = makeProc({ delayMs: 60_000 });
      const proc2 = makeProc({ delayMs: 60_000 });
      spawnMock.mockReturnValueOnce(proc1).mockReturnValueOnce(proc2);
      const { GooseWorkerPool } = await import("../../orchestration/goose-worker.js");

      const pool = new GooseWorkerPool({ db, provider: "ollama", model: "qwen3.5:4b", workerId: "pool-test" });
      pool.spawn(makeTask({ id: "t1" }));
      pool.spawn(makeTask({ id: "t2" }));

      expect(pool.getActiveCount()).toBe(2);

      await pool.shutdown();

      expect(proc1.kill).toHaveBeenCalledWith("SIGTERM");
      expect(proc2.kill).toHaveBeenCalledWith("SIGTERM");
      expect(pool.getActiveCount()).toBe(0);
    });
  });
});
