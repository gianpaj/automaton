/**
 * SubprocessWorkerPool — Abstract Base
 *
 * Shared lifecycle for worker pools that execute tasks by spawning an external
 * CLI subprocess (Pi, Goose, or any future tool). Handles:
 *   - Worker registry (spawn / hasWorker / getActiveCount)
 *   - Graceful shutdown (SIGTERM → wait → SIGKILL)
 *   - Subprocess lifetime: timeout, abort, error and exit handling
 *   - Task graph wiring (completeTask / failTask)
 *   - Wake event emission on task completion or failure
 *   - Task prompt construction with dependency output injection
 *   - Message content normalisation (string | content-block array → string)
 *
 * Subclasses implement three things only:
 *   1. buildInvocation(prompt)  — returns the cmd + args to spawn
 *   2. parseOutput(stdout)      — extracts the result string from raw stdout
 *   3. checkAvailability()      — logs a warning if the binary is missing
 */

import { spawn, exec as execCb } from "node:child_process";
import type { ChildProcess } from "node:child_process";
import { ulid } from "ulid";
import { createLogger } from "../observability/logger.js";
import { completeTask, failTask } from "./task-graph.js";
import type { TaskNode, TaskResult } from "./task-graph.js";
import { getTaskById, insertWakeEvent } from "../state/database.js";
import type { Database } from "better-sqlite3";

const DEFAULT_TIMEOUT_MS = 5 * 60_000;

export interface SubprocessWorkerConfig {
  db: Database;
  workerId: string;
  /**
   * Maximum number of concurrent subprocess workers. Defaults to 2.
   * Subclasses that want capacity limiting should enforce this in spawn().
   */
  maxWorkers?: number;
}

interface ActiveWorker {
  process: ChildProcess;
  taskId: string;
  abortController: AbortController;
}

export abstract class SubprocessWorkerPool {
  protected readonly activeWorkers = new Map<string, ActiveWorker>();
  protected readonly logger = createLogger("orchestration.subprocess-worker");

  constructor(protected readonly config: SubprocessWorkerConfig) {}

  // ─── Abstract interface ────────────────────────────────────────

  /** Human-readable tag used in log lines, e.g. "PI-WORKER" or "GOOSE-WORKER". */
  protected abstract get logTag(): string;

  /**
   * Return the command and arguments needed to run one task.
   * The `prompt` argument contains the full task description.
   */
  protected abstract buildInvocation(prompt: string): { cmd: string; args: string[] };

  /**
   * Extract the result string from the subprocess stdout.
   * Called only on exit code 0. Should never throw — return a fallback string
   * if the output cannot be parsed.
   */
  protected abstract parseOutput(stdout: string, workerId: string): string;

  /**
   * Check that the required binary exists and emit an appropriate warning if not.
   * Called once at construction time (fire-and-forget).
   */
  protected abstract checkAvailability(): void;

  // ─── Public API ────────────────────────────────────────────────

  spawn(task: TaskNode): { address: string; name: string; sandboxId: string } {
    const workerId = `${this.logTag.toLowerCase()}-${ulid()}`;
    // Use the first word of logTag as the name prefix: "PI-WORKER" → "pi"
    const namePrefix = this.logTag.split("-")[0].toLowerCase();
    const workerName = `${namePrefix}-${task.agentRole ?? "generalist"}-${workerId.slice(-6)}`;
    const address = `local://${workerId}`;
    const abortController = new AbortController();

    this.runWorker(workerId, task, abortController)
      .catch((error) => {
        // runWorker calls failTask before rejecting on all known paths.
        // This catch is a last-resort safety net — log only, never double-call failTask.
        this.logger.error(
          `[${this.logTag}] Worker crashed unexpectedly`,
          error instanceof Error ? error : new Error(String(error)),
          { workerId, taskId: task.id },
        );
      })
      .finally(() => {
        this.activeWorkers.delete(workerId);
      });

    this.logger.info(`[${this.logTag} ${workerId}] Spawned for task "${task.title}" (${task.id})`);
    return { address, name: workerName, sandboxId: workerId };
  }

  hasWorker(addressOrId: string): boolean {
    const id = addressOrId.replace("local://", "");
    return this.activeWorkers.has(id);
  }

  getActiveCount(): number {
    return this.activeWorkers.size;
  }

  async shutdown(): Promise<void> {
    for (const [, worker] of this.activeWorkers) {
      worker.abortController.abort();
      worker.process.kill("SIGTERM");
    }
    await new Promise<void>((resolve) => setTimeout(resolve, 500));
    for (const [, worker] of this.activeWorkers) {
      if (!worker.process.killed) worker.process.kill("SIGKILL");
    }
    this.activeWorkers.clear();
  }

  // ─── Shared internals ─────────────────────────────────────────

  private async runWorker(
    workerId: string,
    task: TaskNode,
    abortController: AbortController,
  ): Promise<void> {
    const timeoutMs = task.metadata?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const prompt = this.buildTaskPrompt(task);
    const startedAt = Date.now();

    this.logger.info(`[${this.logTag} ${workerId}] Starting — timeout=${timeoutMs}ms`);

    const { cmd, args } = this.buildInvocation(prompt);

    await new Promise<void>((resolve, reject) => {
      const proc = spawn(cmd, args, {
        env: process.env,
        stdio: ["ignore", "pipe", "pipe"],
      });

      this.activeWorkers.set(workerId, { process: proc, taskId: task.id, abortController });

      const stdoutChunks: Buffer[] = [];
      const stderrChunks: Buffer[] = [];

      proc.stdout?.on("data", (chunk: Buffer) => stdoutChunks.push(chunk));
      proc.stderr?.on("data", (chunk: Buffer) => {
        stderrChunks.push(chunk);
        for (const line of chunk.toString().split("\n").filter(Boolean)) {
          this.logger.info(`[${this.logTag} ${workerId}] stderr: ${line}`);
        }
      });

      const timeoutHandle = setTimeout(() => {
        this.logger.warn(`[${this.logTag} ${workerId}] Timed out after ${timeoutMs}ms`);
        proc.kill("SIGTERM");
        setTimeout(() => { if (!proc.killed) proc.kill("SIGKILL"); }, 3000);
        try {
          failTask(this.config.db, task.id, `Worker timed out after ${timeoutMs}ms`, true);
          insertWakeEvent(this.config.db, this.logTag.toLowerCase(), `Task timed out: ${task.title}`);
        } catch { /* terminal */ }
        reject(new Error(`Worker timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      abortController.signal.addEventListener("abort", () => {
        clearTimeout(timeoutHandle);
        proc.kill("SIGTERM");
        try {
          failTask(this.config.db, task.id, "Worker aborted", false);
        } catch { /* terminal */ }
        reject(new Error("Worker aborted"));
      });

      proc.on("error", (err) => {
        clearTimeout(timeoutHandle);
        try {
          failTask(this.config.db, task.id, `Process error: ${err.message}`, true);
          insertWakeEvent(this.config.db, this.logTag.toLowerCase(), `Task error: ${task.title}`);
        } catch { /* terminal */ }
        reject(err);
      });

      proc.on("exit", (code) => {
        clearTimeout(timeoutHandle);

        if (abortController.signal.aborted) { resolve(); return; }

        const stdout = Buffer.concat(stdoutChunks).toString();
        const duration = Date.now() - startedAt;

        if (code === 0) {
          const output = this.parseOutput(stdout, workerId);
          this.logger.info(`[${this.logTag} ${workerId}] Completed in ${duration}ms — ${output.slice(0, 200)}`);

          const result: TaskResult = { success: true, output, artifacts: [], costCents: 0, duration };
          try {
            completeTask(this.config.db, task.id, result);
            insertWakeEvent(this.config.db, this.logTag.toLowerCase(), `Task completed: ${task.title}`);
          } catch (err) {
            this.logger.warn(`[${this.logTag} ${workerId}] Failed to mark task complete`, {
              error: err instanceof Error ? err.message : String(err),
            });
          }
          resolve();
        } else {
          const stderr = Buffer.concat(stderrChunks).toString();
          const errorMsg = `Exited with code ${code}. stderr: ${stderr.slice(0, 500)}`;
          this.logger.warn(`[${this.logTag} ${workerId}] Failed: ${errorMsg}`);
          try {
            failTask(this.config.db, task.id, errorMsg, true);
            insertWakeEvent(this.config.db, this.logTag.toLowerCase(), `Task failed: ${task.title}`);
          } catch { /* terminal */ }
          reject(new Error(errorMsg));
        }
      });
    });
  }

  // ─── Shared helpers ───────────────────────────────────────────

  protected buildTaskPrompt(task: TaskNode): string {
    const lines = [
      `Task: ${task.title}`,
      `Description: ${task.description}`,
      `Role: ${task.agentRole ?? "generalist"}`,
      `Task ID: ${task.id}`,
    ];

    if (task.dependencies?.length) {
      const depSection = this.buildDependencySection(task.dependencies);
      lines.push("", depSection);
    }

    lines.push(
      "",
      "INSTRUCTIONS:",
      "- You MUST actually execute this task using your tools (shell commands, file writes, etc.).",
      "- Do NOT just describe what you would do — actually do it.",
      "- Write files to disk, run shell commands, verify the results.",
      "- Provide a concise final summary of what was accomplished.",
    );

    return lines.join("\n");
  }

  /**
   * Look up each dependency's completed result from the DB and format them
   * as a readable section so the worker can build on prior task outputs.
   */
  private buildDependencySection(depIds: string[]): string {
    const MAX_OUTPUT_CHARS = 2000;
    const parts: string[] = ["Dependency Results (completed tasks this task builds on):"];

    for (const id of depIds) {
      const row = getTaskById(this.config.db, id);
      if (!row) {
        parts.push(`\n### ${id}\n(task not found)`);
        continue;
      }

      const result = row.result as TaskResult | null;
      if (!result?.output) {
        parts.push(`\n### ${row.title} (${id})\n(no output recorded)`);
        continue;
      }

      const output = result.output.length > MAX_OUTPUT_CHARS
        ? result.output.slice(0, MAX_OUTPUT_CHARS) + `\n[…truncated: ${result.output.length - MAX_OUTPUT_CHARS} chars omitted]`
        : result.output;

      parts.push(`\n### ${row.title} (${id})\n${output}`);
    }

    return parts.join("\n");
  }

  /** Find the last assistant message in a messages array and return its text. */
  protected extractLastAssistantText(messages: unknown[], workerId: string): string {
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i] as { role?: string; content?: unknown };
      if (msg?.role !== "assistant") continue;
      const text = this.contentToText(msg.content);
      if (text) return text;
    }
    this.logger.warn(`[${this.logTag} ${workerId}] No assistant message found`);
    return "Task completed (no assistant message found).";
  }

  /**
   * Normalise a message `content` field to a plain string.
   * Handles both raw strings and typed content-block arrays:
   *   [{ type: "text", text: "..." }, ...]
   */
  protected contentToText(content: unknown): string {
    if (typeof content === "string") return content.trim();
    if (Array.isArray(content)) {
      return content
        .filter((b: unknown) => (b as any)?.type === "text" && typeof (b as any)?.text === "string")
        .map((b: unknown) => (b as any).text as string)
        .join("\n")
        .trim();
    }
    return "";
  }

  /** Convenience wrapper for fire-and-forget availability checks. */
  protected execCheck(cmd: string, onMissing: () => void, onFound?: () => void): void {
    execCb(cmd, (error) => {
      if (error) onMissing();
      else onFound?.();
    });
  }
}
