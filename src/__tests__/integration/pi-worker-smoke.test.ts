/**
 * Pi Worker Integration Smoke Test
 *
 * Spawns a real Pi subprocess against a real LLM to verify the full
 * PiWorkerPool → task-graph pipeline end-to-end.
 *
 * Skipped automatically unless ANTHROPIC_API_KEY (or OPENAI_API_KEY /
 * GEMINI_API_KEY) is present in the environment, so it never blocks CI
 * without credentials.
 *
 * Run manually:
 *   ANTHROPIC_API_KEY=sk-ant-... npx vitest run src/__tests__/integration/pi-worker-smoke.test.ts
 */

import { describe, it, expect, beforeAll } from "vitest";
import { createDatabase } from "../../state/database.js";
import { getTaskById } from "../../state/database.js";
import { PiWorkerPool } from "../../orchestration/pi-worker.js";
import { createGoal, decomposeGoal, getReadyTasks, completeTask } from "../../orchestration/task-graph.js";
import type { TaskNode } from "../../orchestration/task-graph.js";
import os from "node:os";
import path from "node:path";
import fs from "node:fs";

// ─── Skip guard ───────────────────────────────────────────────────────────────

const PROVIDER = process.env.ANTHROPIC_API_KEY ? "anthropic"
  : process.env.OPENAI_API_KEY ? "openai"
  : process.env.GEMINI_API_KEY ? "google"
  : null;

const MODEL = process.env.ANTHROPIC_API_KEY ? "claude-haiku-4-5-20251001"
  : process.env.OPENAI_API_KEY ? "gpt-4o-mini"
  : process.env.GEMINI_API_KEY ? "gemini-2.0-flash"
  : "";

const itIfCreds = PROVIDER ? it : it.skip;

// ─── Helpers ──────────────────────────────────────────────────────────────────

function makeTmpDb() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-smoke-"));
  return createDatabase(path.join(dir, "test.db"));
}

async function pollUntilDone(db: ReturnType<typeof makeTmpDb>, taskId: string, timeoutMs = 60_000) {
  const deadline = Date.now() + timeoutMs;
  let row = getTaskById(db.raw, taskId);
  while (row?.status !== "completed" && row?.status !== "failed" && Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 500));
    row = getTaskById(db.raw, taskId);
  }
  return row;
}

// ─── Tests ────────────────────────────────────────────────────────────────────

describe("PiWorkerPool integration smoke test", () => {
  let db: ReturnType<typeof makeTmpDb>;

  beforeAll(() => {
    if (!PROVIDER) return;
    db = makeTmpDb();
  });

  itIfCreds(
    "completes a trivial task and writes result to the DB",
    async () => {
      const goal = createGoal(db.raw, "Smoke test", "Reply with SMOKE_DONE");
      decomposeGoal(db.raw, goal.id, [{
        parentId: null,
        goalId: goal.id,
        title: "Echo check",
        description: "Reply with exactly the word SMOKE_DONE and nothing else.",
        status: "pending",
        assignedTo: null,
        agentRole: "generalist",
        priority: 50,
        dependencies: [],
        result: null,
      }]);
      const [task] = getReadyTasks(db.raw);

      const pool = new PiWorkerPool({
        db: db.raw,
        provider: PROVIDER!,
        model: MODEL,
        workerId: "smoke-test",
      });
      pool.spawn(task);

      const row = await pollUntilDone(db, task.id);
      await pool.shutdown();

      expect(row?.status).toBe("completed");
      const result = row?.result as { output: string } | null;
      expect(result?.output.toUpperCase()).toContain("SMOKE_DONE");
    },
    70_000,
  );

  itIfCreds(
    "injects prior task output into dependent task prompt",
    async () => {
      const goal = createGoal(db.raw, "Dep injection test", "Test dependency passing");

      // Insert task1 already completed with known output
      decomposeGoal(db.raw, goal.id, [{
        parentId: null,
        goalId: goal.id,
        title: "Produce token",
        description: "Produce the token TOKEN_XYZ_789",
        status: "pending",
        assignedTo: null,
        agentRole: "generalist",
        priority: 50,
        dependencies: [],
        result: null,
      }]);
      const [task1] = getReadyTasks(db.raw);

      // Manually complete task1 with a known output (no LLM call needed)
      completeTask(db.raw, task1.id, {
        success: true,
        output: "The token is: TOKEN_XYZ_789",
        artifacts: [],
        costCents: 0,
        duration: 1,
      });

      // Insert task2 that depends on task1
      decomposeGoal(db.raw, goal.id, [{
        parentId: null,
        goalId: goal.id,
        title: "Echo token",
        description: "Look at the dependency result and echo back the token that starts with TOKEN_.",
        status: "pending",
        assignedTo: null,
        agentRole: "generalist",
        priority: 50,
        dependencies: [task1.id],
        result: null,
      }]);
      const ready = getReadyTasks(db.raw);
      const task2 = ready.find((t) => t.title === "Echo token")!;

      const pool = new PiWorkerPool({
        db: db.raw,
        provider: PROVIDER!,
        model: MODEL,
        workerId: "smoke-dep-test",
      });
      pool.spawn(task2);

      const row2 = await pollUntilDone(db, task2.id, 60_000);
      await pool.shutdown();

      expect(row2?.status).toBe("completed");
      const result2 = row2?.result as { output: string } | null;
      expect(result2?.output.toUpperCase()).toContain("TOKEN_XYZ_789");
    },
    70_000,
  );
});
