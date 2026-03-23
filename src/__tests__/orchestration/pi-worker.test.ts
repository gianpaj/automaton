import { EventEmitter } from "node:events";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type BetterSqlite3 from "better-sqlite3";
import type { TaskNode } from "../../orchestration/task-graph.js";

// ─── Mocks ──────────────────────────────────────────────────────────────────

vi.mock("node:child_process", () => ({
  spawn: vi.fn(),
  exec: vi.fn((_cmd: string, cb: (err: Error | null) => void) => {
    cb(null);
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

vi.mock("../../state/database.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../state/database.js")>();
  return { ...actual, getTaskById: vi.fn() };
});

// ─── Helpers ────────────────────────────────────────────────────────────────

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
    assignedTo: "pi-worker-abc",
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

function makeDb(): BetterSqlite3.Database {
  return {} as unknown as BetterSqlite3.Database;
}

// ─── Tests ──────────────────────────────────────────────────────────────────

describe("PiWorkerPool", () => {
  let db: BetterSqlite3.Database;
  let spawnMock: ReturnType<typeof vi.fn>;
  let completeTask: ReturnType<typeof vi.fn>;
  let failTask: ReturnType<typeof vi.fn>;
  let getTaskById: ReturnType<typeof vi.fn>;

  beforeEach(async () => {
    db = makeDb();
    vi.clearAllMocks();

    const childProcess = await import("node:child_process");
    spawnMock = childProcess.spawn as unknown as ReturnType<typeof vi.fn>;

    const taskGraph = await import("../../orchestration/task-graph.js");
    completeTask = taskGraph.completeTask as unknown as ReturnType<typeof vi.fn>;
    failTask = taskGraph.failTask as unknown as ReturnType<typeof vi.fn>;

    const database = await import("../../state/database.js");
    getTaskById = database.getTaskById as unknown as ReturnType<typeof vi.fn>;
    getTaskById.mockReturnValue(undefined); // default: dep not found
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // ── spawn() shape ─────────────────────────────────────────────────────────

  describe("spawn()", () => {
    it("returns correct address/name/sandboxId shape", async () => {
      spawnMock.mockReturnValue(makeProc());
      const { PiWorkerPool } = await import("../../orchestration/pi-worker.js");

      const pool = new PiWorkerPool({ db, provider: "anthropic", model: "claude-haiku-4-5-20251001", workerId: "pool-test", credentialsVerified: true });
      const result = pool.spawn(makeTask());

      expect(result.address).toMatch(/^local:\/\/pi-worker-/);
      expect(result.name).toMatch(/^pi-generalist-/);
      expect(result.sandboxId).toMatch(/^pi-worker-/);
      expect(result.sandboxId).toBe(result.address.replace("local://", ""));

      await pool.shutdown();
    });

    it("includes provider and model in CLI args when configured", async () => {
      spawnMock.mockReturnValue(makeProc());
      const { PiWorkerPool } = await import("../../orchestration/pi-worker.js");

      const pool = new PiWorkerPool({ db, provider: "anthropic", model: "claude-haiku-4-5-20251001", workerId: "pool-test", credentialsVerified: true });
      pool.spawn(makeTask());
      await new Promise((r) => setImmediate(r));

      const allArgs: string[] = spawnMock.mock.calls[0][1];
      expect(allArgs).toContain("--provider");
      expect(allArgs).toContain("anthropic");
      expect(allArgs).toContain("--model");
      expect(allArgs).toContain("claude-haiku-4-5-20251001");

      await pool.shutdown();
    });

    it("omits --provider and --model args when config has no provider", async () => {
      spawnMock.mockReturnValue(makeProc());
      const { PiWorkerPool } = await import("../../orchestration/pi-worker.js");

      const pool = new PiWorkerPool({ db, workerId: "pool-test", credentialsVerified: true });
      pool.spawn(makeTask());
      await new Promise((r) => setImmediate(r));

      const allArgs: string[] = spawnMock.mock.calls[0][1];
      expect(allArgs).not.toContain("--provider");
      expect(allArgs).not.toContain("--model");

      await pool.shutdown();
    });

    it("uses --mode json --no-session -p flags", async () => {
      spawnMock.mockReturnValue(makeProc());
      const { PiWorkerPool } = await import("../../orchestration/pi-worker.js");

      const pool = new PiWorkerPool({ db, provider: "anthropic", model: "claude-haiku-4-5-20251001", workerId: "pool-test", credentialsVerified: true });
      pool.spawn(makeTask());
      await new Promise((r) => setImmediate(r));

      const allArgs: string[] = spawnMock.mock.calls[0][1];
      expect(allArgs).toContain("--mode");
      expect(allArgs).toContain("json");
      expect(allArgs).toContain("--no-session");
      expect(allArgs).toContain("-p");

      await pool.shutdown();
    });

    it("includes task title and description in the -p prompt", async () => {
      spawnMock.mockReturnValue(makeProc());
      const { PiWorkerPool } = await import("../../orchestration/pi-worker.js");

      const task = makeTask({ title: "Build API", description: "Create REST endpoints" });
      const pool = new PiWorkerPool({ db, provider: "anthropic", model: "claude-haiku-4-5-20251001", workerId: "pool-test", credentialsVerified: true });
      pool.spawn(task);
      await new Promise((r) => setImmediate(r));

      const allArgs: string[] = spawnMock.mock.calls[0][1];
      const pIndex = allArgs.indexOf("-p");
      const prompt = allArgs[pIndex + 1];
      expect(prompt).toContain("Build API");
      expect(prompt).toContain("Create REST endpoints");

      await pool.shutdown();
    });
  });

  // ── hasWorker() / getActiveCount() ───────────────────────────────────────

  describe("hasWorker() / getActiveCount()", () => {
    it("hasWorker() returns true immediately after spawn, before process exits", async () => {
      const proc = makeProc({ delayMs: 50 });
      spawnMock.mockReturnValue(proc);
      const { PiWorkerPool } = await import("../../orchestration/pi-worker.js");

      const pool = new PiWorkerPool({ db, provider: "anthropic", model: "claude-haiku-4-5-20251001", workerId: "pool-test", credentialsVerified: true });
      const { address } = pool.spawn(makeTask());

      expect(pool.hasWorker(address)).toBe(true);
      expect(pool.getActiveCount()).toBe(1);

      await pool.shutdown();
    });

    it("hasWorker() accepts raw workerId without the local:// prefix", async () => {
      spawnMock.mockReturnValue(makeProc({ delayMs: 50 }));
      const { PiWorkerPool } = await import("../../orchestration/pi-worker.js");

      const pool = new PiWorkerPool({ db, provider: "anthropic", model: "claude-haiku-4-5-20251001", workerId: "pool-test", credentialsVerified: true });
      const { address } = pool.spawn(makeTask());
      const rawId = address.replace("local://", "");

      expect(pool.hasWorker(rawId)).toBe(true);

      await pool.shutdown();
    });

    it("hasWorker() returns false after process exits", async () => {
      spawnMock.mockReturnValue(makeProc({ exitCode: 0, stdout: "done" }));
      const { PiWorkerPool } = await import("../../orchestration/pi-worker.js");

      const pool = new PiWorkerPool({ db, provider: "anthropic", model: "claude-haiku-4-5-20251001", workerId: "pool-test", credentialsVerified: true });
      const { address } = pool.spawn(makeTask());

      for (let i = 0; i < 20; i++) {
        await new Promise((r) => setImmediate(r));
      }

      expect(pool.hasWorker(address)).toBe(false);
      expect(pool.getActiveCount()).toBe(0);
    });

    it("tracks multiple concurrent workers", async () => {
      spawnMock.mockReturnValue(makeProc({ delayMs: 50 }));
      const { PiWorkerPool } = await import("../../orchestration/pi-worker.js");

      const pool = new PiWorkerPool({ db, provider: "anthropic", model: "claude-haiku-4-5-20251001", workerId: "pool-test", credentialsVerified: true });
      pool.spawn(makeTask({ id: "t1" }));
      pool.spawn(makeTask({ id: "t2" }));
      pool.spawn(makeTask({ id: "t3" }));

      expect(pool.getActiveCount()).toBe(3);

      await pool.shutdown();
    });
  });

  // ── Success path ──────────────────────────────────────────────────────────

  describe("successful completion", () => {
    it("calls completeTask with success=true on exit code 0 (plain text output)", async () => {
      spawnMock.mockReturnValue(makeProc({ exitCode: 0, stdout: "Task done!" }));
      const { PiWorkerPool } = await import("../../orchestration/pi-worker.js");

      const pool = new PiWorkerPool({ db, provider: "anthropic", model: "claude-haiku-4-5-20251001", workerId: "pool-test", credentialsVerified: true });
      const task = makeTask();
      pool.spawn(task);

      for (let i = 0; i < 20; i++) await new Promise((r) => setImmediate(r));

      expect(completeTask).toHaveBeenCalledWith(
        db,
        task.id,
        expect.objectContaining({ success: true }),
      );
    });

    it("parses Pi --mode json JSONL: agent_end event with text content blocks", async () => {
      const agentEnd = JSON.stringify({
        type: "agent_end",
        messages: [
          { role: "user", content: [{ type: "text", text: "Do it" }] },
          {
            role: "assistant",
            content: [
              { type: "thinking", thinking: "Let me think..." },
              { type: "text", text: "Here is my result." },
            ],
          },
        ],
      });
      spawnMock.mockReturnValue(makeProc({ exitCode: 0, stdout: agentEnd }));
      const { PiWorkerPool } = await import("../../orchestration/pi-worker.js");

      const pool = new PiWorkerPool({ db, provider: "anthropic", model: "claude-haiku-4-5-20251001", workerId: "pool-test", credentialsVerified: true });
      pool.spawn(makeTask());

      for (let i = 0; i < 20; i++) await new Promise((r) => setImmediate(r));

      expect(completeTask).toHaveBeenCalledWith(
        db,
        expect.any(String),
        expect.objectContaining({ output: "Here is my result." }),
      );
    });

    it("parses multiple JSONL lines and uses last agent_end event", async () => {
      const progressEvent = JSON.stringify({ type: "turn_start" });
      const agentEnd = JSON.stringify({
        type: "agent_end",
        messages: [{ role: "assistant", content: [{ type: "text", text: "Final answer." }] }],
      });
      const stdout = [progressEvent, agentEnd].join("\n");
      spawnMock.mockReturnValue(makeProc({ exitCode: 0, stdout }));
      const { PiWorkerPool } = await import("../../orchestration/pi-worker.js");

      const pool = new PiWorkerPool({ db, provider: "anthropic", model: "claude-haiku-4-5-20251001", workerId: "pool-test", credentialsVerified: true });
      pool.spawn(makeTask());

      for (let i = 0; i < 20; i++) await new Promise((r) => setImmediate(r));

      expect(completeTask).toHaveBeenCalledWith(
        db,
        expect.any(String),
        expect.objectContaining({ output: "Final answer." }),
      );
    });

    it("falls back to raw stdout when no agent_end event is present", async () => {
      spawnMock.mockReturnValue(makeProc({ exitCode: 0, stdout: "Plain text result" }));
      const { PiWorkerPool } = await import("../../orchestration/pi-worker.js");

      const pool = new PiWorkerPool({ db, provider: "anthropic", model: "claude-haiku-4-5-20251001", workerId: "pool-test", credentialsVerified: true });
      pool.spawn(makeTask());

      for (let i = 0; i < 20; i++) await new Promise((r) => setImmediate(r));

      expect(completeTask).toHaveBeenCalledWith(
        db,
        expect.any(String),
        expect.objectContaining({ output: "Plain text result" }),
      );
    });
  });

  // ── Failure paths ─────────────────────────────────────────────────────────

  describe("failure handling", () => {
    it("calls failTask on non-zero exit code", async () => {
      spawnMock.mockReturnValue(makeProc({ exitCode: 1, stderr: "something went wrong" }));
      const { PiWorkerPool } = await import("../../orchestration/pi-worker.js");

      const pool = new PiWorkerPool({ db, provider: "anthropic", model: "claude-haiku-4-5-20251001", workerId: "pool-test", credentialsVerified: true });
      const task = makeTask();
      pool.spawn(task);

      for (let i = 0; i < 20; i++) await new Promise((r) => setImmediate(r));

      expect(failTask).toHaveBeenCalledWith(
        db,
        task.id,
        expect.stringContaining("code 1"),
        true,
      );
      expect(completeTask).not.toHaveBeenCalled();
    });

    it("calls failTask on process error event", async () => {
      const err = new Error("ENOENT: pi not found");
      spawnMock.mockReturnValue(makeProc({ errorToEmit: err }));
      const { PiWorkerPool } = await import("../../orchestration/pi-worker.js");

      const pool = new PiWorkerPool({ db, provider: "anthropic", model: "claude-haiku-4-5-20251001", workerId: "pool-test", credentialsVerified: true });
      const task = makeTask();
      pool.spawn(task);

      for (let i = 0; i < 20; i++) await new Promise((r) => setImmediate(r));

      expect(failTask).toHaveBeenCalledWith(
        db,
        task.id,
        expect.stringContaining("ENOENT"),
        true,
      );
    });

    it("does not double-call failTask on process error", async () => {
      const err = new Error("spawn failed");
      spawnMock.mockReturnValue(makeProc({ errorToEmit: err }));
      const { PiWorkerPool } = await import("../../orchestration/pi-worker.js");

      const pool = new PiWorkerPool({ db, provider: "anthropic", model: "claude-haiku-4-5-20251001", workerId: "pool-test", credentialsVerified: true });
      pool.spawn(makeTask());

      for (let i = 0; i < 20; i++) await new Promise((r) => setImmediate(r));

      expect(failTask).toHaveBeenCalledTimes(1);
    });
  });

  // ── Timeout ───────────────────────────────────────────────────────────────

  describe("timeout", () => {
    it("calls failTask and kills the process when timeout expires", async () => {
      // Use a real short timer — the task metadata sets timeoutMs to 50ms
      const proc = makeProc({ delayMs: 2_000 }); // Process takes 2s — will time out first
      spawnMock.mockReturnValue(proc);
      const { PiWorkerPool } = await import("../../orchestration/pi-worker.js");

      const pool = new PiWorkerPool({ db, provider: "anthropic", model: "claude-haiku-4-5-20251001", workerId: "pool-test", credentialsVerified: true });
      pool.spawn(makeTask({ metadata: { timeoutMs: 50 } as any }));

      await new Promise((r) => setTimeout(r, 100)); // wait for timeout

      expect(proc.kill).toHaveBeenCalled();
      expect(failTask).toHaveBeenCalledWith(
        db,
        expect.any(String),
        expect.stringContaining("timed out"),
        true,
      );
    });
  });

  // ── dependency result injection ───────────────────────────────────────────

  describe("dependency result injection", () => {
    it("includes dependency output in the prompt when the dep has a result", async () => {
      spawnMock.mockReturnValue(makeProc());
      getTaskById.mockReturnValue({
        id: "dep-01",
        title: "Fetch data",
        result: { success: true, output: "The fetched data is: [1, 2, 3]", artifacts: [], costCents: 0, duration: 100 },
      });

      const { PiWorkerPool } = await import("../../orchestration/pi-worker.js");
      const pool = new PiWorkerPool({ db, provider: "anthropic", model: "claude-haiku-4-5-20251001", workerId: "pool-test", credentialsVerified: true });
      pool.spawn(makeTask({ dependencies: ["dep-01"] }));
      await new Promise((r) => setImmediate(r));

      const allArgs: string[] = spawnMock.mock.calls[0][1];
      const prompt = allArgs[allArgs.indexOf("-p") + 1];

      expect(prompt).toContain("Dependency Results");
      expect(prompt).toContain("Fetch data");
      expect(prompt).toContain("The fetched data is: [1, 2, 3]");

      await pool.shutdown();
    });

    it("falls back gracefully when a dependency has no result", async () => {
      spawnMock.mockReturnValue(makeProc());
      getTaskById.mockReturnValue({ id: "dep-02", title: "Missing task", result: null });

      const { PiWorkerPool } = await import("../../orchestration/pi-worker.js");
      const pool = new PiWorkerPool({ db, provider: "anthropic", model: "claude-haiku-4-5-20251001", workerId: "pool-test", credentialsVerified: true });
      pool.spawn(makeTask({ dependencies: ["dep-02"] }));
      await new Promise((r) => setImmediate(r));

      const prompt = spawnMock.mock.calls[0][1][spawnMock.mock.calls[0][1].indexOf("-p") + 1];
      expect(prompt).toContain("Dependency Results");
      expect(prompt).toContain("no output recorded");

      await pool.shutdown();
    });

    it("falls back gracefully when a dependency is not found in the DB", async () => {
      spawnMock.mockReturnValue(makeProc());
      getTaskById.mockReturnValue(undefined);

      const { PiWorkerPool } = await import("../../orchestration/pi-worker.js");
      const pool = new PiWorkerPool({ db, provider: "anthropic", model: "claude-haiku-4-5-20251001", workerId: "pool-test", credentialsVerified: true });
      pool.spawn(makeTask({ dependencies: ["ghost-task"] }));
      await new Promise((r) => setImmediate(r));

      const prompt = spawnMock.mock.calls[0][1][spawnMock.mock.calls[0][1].indexOf("-p") + 1];
      expect(prompt).toContain("Dependency Results");
      expect(prompt).toContain("task not found");

      await pool.shutdown();
    });

    it("truncates dependency output longer than 2000 chars", async () => {
      spawnMock.mockReturnValue(makeProc());
      const longOutput = "x".repeat(3000);
      getTaskById.mockReturnValue({
        id: "dep-03",
        title: "Big output task",
        result: { success: true, output: longOutput, artifacts: [], costCents: 0, duration: 50 },
      });

      const { PiWorkerPool } = await import("../../orchestration/pi-worker.js");
      const pool = new PiWorkerPool({ db, provider: "anthropic", model: "claude-haiku-4-5-20251001", workerId: "pool-test", credentialsVerified: true });
      pool.spawn(makeTask({ dependencies: ["dep-03"] }));
      await new Promise((r) => setImmediate(r));

      const prompt = spawnMock.mock.calls[0][1][spawnMock.mock.calls[0][1].indexOf("-p") + 1];
      expect(prompt).toContain("truncated");
      expect(prompt).not.toContain(longOutput); // full 3000-char string must not appear

      await pool.shutdown();
    });

    it("omits dependency section entirely when task has no dependencies", async () => {
      spawnMock.mockReturnValue(makeProc());

      const { PiWorkerPool } = await import("../../orchestration/pi-worker.js");
      const pool = new PiWorkerPool({ db, provider: "anthropic", model: "claude-haiku-4-5-20251001", workerId: "pool-test", credentialsVerified: true });
      pool.spawn(makeTask({ dependencies: [] }));
      await new Promise((r) => setImmediate(r));

      const prompt = spawnMock.mock.calls[0][1][spawnMock.mock.calls[0][1].indexOf("-p") + 1];
      expect(prompt).not.toContain("Dependency Results");

      await pool.shutdown();
    });
  });

  // ── shutdown() ────────────────────────────────────────────────────────────

  describe("shutdown()", () => {
    it("kills all active workers and clears the pool", async () => {
      const proc1 = makeProc({ delayMs: 5_000 });
      const proc2 = makeProc({ delayMs: 5_000 });
      spawnMock.mockReturnValueOnce(proc1).mockReturnValueOnce(proc2);
      const { PiWorkerPool } = await import("../../orchestration/pi-worker.js");

      const pool = new PiWorkerPool({ db, provider: "anthropic", model: "claude-haiku-4-5-20251001", workerId: "pool-test", credentialsVerified: true });
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
