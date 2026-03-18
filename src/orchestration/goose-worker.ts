/**
 * Goose Worker Pool
 *
 * Replaces LocalWorkerPool with Goose (block/goose) subprocess-based workers.
 * Each worker invokes `goose run --no-session` as a subprocess, allowing use of
 * Ollama (free local inference) or any other Goose-supported provider.
 */

import { ulid } from "ulid";
import { spawn, exec as execCb } from "node:child_process";
import type { ChildProcess } from "node:child_process";
import { createLogger } from "../observability/logger.js";
import { completeTask, failTask } from "./task-graph.js";
import type { TaskNode, TaskResult } from "./task-graph.js";
import type { Database } from "better-sqlite3";

const logger = createLogger("orchestration.goose-worker");

const DEFAULT_TIMEOUT_MS = 5 * 60_000;

interface GooseWorkerConfig {
  db: Database;
  /** Override Goose's configured provider. Omit to use whatever `goose configure` set. */
  provider?: string;
  /** Override Goose's configured model. Omit to use whatever `goose configure` set. */
  model?: string;
  workerId: string;
}

interface ActiveWorker {
  process: ChildProcess;
  taskId: string;
  abortController: AbortController;
}

export class GooseWorkerPool {
  private activeWorkers = new Map<string, ActiveWorker>();
  private gooseAvailable: boolean | null = null;

  constructor(private readonly config: GooseWorkerConfig) {
    this.checkGooseAvailability();
  }

  private checkGooseAvailability(): void {
    execCb("which goose", (error) => {
      if (error) {
        this.gooseAvailable = false;
        logger.warn(
          "[GOOSE-WORKER] Goose not found in PATH. " +
          "Install with: curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh | CONFIGURE=false bash"
        );
      } else {
        this.gooseAvailable = true;
        const providerInfo = this.config.provider
          ? `provider=${this.config.provider} model=${this.config.model}`
          : "provider=<goose default>";
        logger.info(`[GOOSE-WORKER] Goose available. ${providerInfo}`);
      }
    });
  }

  spawn(task: TaskNode): { address: string; name: string; sandboxId: string } {
    const workerId = `goose-worker-${ulid()}`;
    const workerName = `goose-${task.agentRole ?? "generalist"}-${workerId.slice(-6)}`;
    const address = `local://${workerId}`;
    const abortController = new AbortController();

    this.runWorker(workerId, task, abortController)
      .catch((error) => {
        // runWorker calls failTask internally before rejecting for all known error
        // paths. This catch is a last-resort safety net for unexpected throws —
        // log only, do not call failTask again (it would error on an already-terminal task).
        logger.error(
          "[GOOSE-WORKER] Worker crashed unexpectedly",
          error instanceof Error ? error : new Error(String(error)),
          { workerId, taskId: task.id }
        );
      })
      .finally(() => {
        this.activeWorkers.delete(workerId);
      });

    logger.info(`[GOOSE-WORKER ${workerId}] Spawned for task "${task.title}" (${task.id})`);
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
    // Give processes a moment to exit gracefully
    await new Promise<void>((resolve) => setTimeout(resolve, 500));
    for (const [, worker] of this.activeWorkers) {
      if (!worker.process.killed) {
        worker.process.kill("SIGKILL");
      }
    }
    this.activeWorkers.clear();
  }

  private async runWorker(
    workerId: string,
    task: TaskNode,
    abortController: AbortController,
  ): Promise<void> {
    const timeoutMs = task.metadata?.timeoutMs || DEFAULT_TIMEOUT_MS;
    const prompt = this.buildTaskPrompt(task);
    const startedAt = Date.now();

    const providerInfo = this.config.provider
      ? `provider=${this.config.provider} model=${this.config.model}`
      : "provider=<goose default>";
    logger.info(`[GOOSE-WORKER ${workerId}] Starting — ${providerInfo} timeout=${timeoutMs}ms`);

    const args = ["run", "--no-session", "--output-format", "json", "-t", prompt];
    if (this.config.provider) args.splice(2, 0, "--provider", this.config.provider);
    if (this.config.model) args.splice(args.indexOf("--output-format"), 0, "--model", this.config.model);

    await new Promise<void>((resolve, reject) => {
      const proc = spawn("goose", args, {
        env: process.env,
        stdio: ["ignore", "pipe", "pipe"],
      });

      // Store in active workers map so hasWorker() works
      this.activeWorkers.set(workerId, {
        process: proc,
        taskId: task.id,
        abortController,
      });

      const stdoutChunks: Buffer[] = [];
      const stderrChunks: Buffer[] = [];

      proc.stdout?.on("data", (chunk: Buffer) => {
        stdoutChunks.push(chunk);
      });

      proc.stderr?.on("data", (chunk: Buffer) => {
        stderrChunks.push(chunk);
        // Log stderr lines for visibility
        const lines = chunk.toString().split("\n").filter(Boolean);
        for (const line of lines) {
          logger.info(`[GOOSE-WORKER ${workerId}] stderr: ${line}`);
        }
      });

      // Timeout handling
      const timeoutHandle = setTimeout(() => {
        logger.warn(`[GOOSE-WORKER ${workerId}] Timed out after ${timeoutMs}ms`);
        proc.kill("SIGTERM");
        setTimeout(() => { if (!proc.killed) proc.kill("SIGKILL"); }, 3000);
        try {
          failTask(this.config.db, task.id, `Goose worker timed out after ${timeoutMs}ms`, true);
        } catch { /* already in terminal state */ }
        reject(new Error(`Worker timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      // Abort handling
      abortController.signal.addEventListener("abort", () => {
        clearTimeout(timeoutHandle);
        proc.kill("SIGTERM");
        try {
          failTask(this.config.db, task.id, "Goose worker aborted", false);
        } catch { /* already in terminal state */ }
        reject(new Error("Worker aborted"));
      });

      proc.on("error", (err) => {
        clearTimeout(timeoutHandle);
        if (err.message.includes("ENOENT")) {
          logger.error(
            "[GOOSE-WORKER] Goose binary not found. " +
            "Install with: curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh | CONFIGURE=false bash"
          );
        }
        try {
          failTask(this.config.db, task.id, `Goose process error: ${err.message}`, true);
        } catch { /* already in terminal state */ }
        reject(err);
      });

      proc.on("exit", (code) => {
        clearTimeout(timeoutHandle);

        if (abortController.signal.aborted) {
          resolve();
          return;
        }

        const stdout = Buffer.concat(stdoutChunks).toString();
        const duration = Date.now() - startedAt;

        if (code === 0) {
          const output = this.parseGooseOutput(stdout, workerId);
          logger.info(`[GOOSE-WORKER ${workerId}] Completed in ${duration}ms — ${output.slice(0, 200)}`);

          const result: TaskResult = {
            success: true,
            output,
            artifacts: [],
            costCents: 0,
            duration,
          };

          try {
            completeTask(this.config.db, task.id, result);
          } catch (err) {
            logger.warn(`[GOOSE-WORKER ${workerId}] Failed to mark task complete`, {
              error: err instanceof Error ? err.message : String(err),
            });
          }
          resolve();
        } else {
          const stderr = Buffer.concat(stderrChunks).toString();
          const errorMsg = `Goose exited with code ${code}. stderr: ${stderr.slice(0, 500)}`;
          logger.warn(`[GOOSE-WORKER ${workerId}] Failed: ${errorMsg}`);
          try {
            failTask(this.config.db, task.id, errorMsg, true);
          } catch { /* already in terminal state */ }
          reject(new Error(errorMsg));
        }
      });
    });
  }

  private parseGooseOutput(stdout: string, workerId: string): string {
    // Goose --output-format json produces: { "messages": [...], "metadata": {...} }
    // Each message has { role: "assistant"|"user"|"tool", content: string|[...] }
    // Older builds or plain text mode fall back to raw stdout.
    const raw = stdout.trim();
    if (!raw) return "Task completed (no output).";

    try {
      const obj = JSON.parse(raw);

      // Primary format: { messages: [...] } (block/goose --output-format json)
      if (obj && Array.isArray(obj.messages)) {
        return this.extractLastAssistantText(obj.messages, workerId);
      }

      // Fallback: bare array of messages
      if (Array.isArray(obj)) {
        return this.extractLastAssistantText(obj, workerId);
      }

      // Fallback: single message object
      if (obj?.role === "assistant") {
        return this.messageContentToText(obj.content);
      }
    } catch {
      // Not JSON — Goose may have run without --output-format json (older version
      // or flag not yet released). Use raw stdout as the result.
      logger.info(`[GOOSE-WORKER ${workerId}] Output is plain text (not JSON), using as-is`);
    }

    return raw;
  }

  /** Extract the text of the last assistant message from a messages array. */
  private extractLastAssistantText(messages: unknown[], workerId: string): string {
    let lastText = "";
    for (const msg of messages) {
      const m = msg as { role?: string; content?: unknown };
      if (m?.role === "assistant") {
        const text = this.messageContentToText(m.content);
        if (text) lastText = text;
      }
    }
    if (lastText) return lastText;
    logger.warn(`[GOOSE-WORKER ${workerId}] No assistant message found in JSON output`);
    return "Task completed (no assistant message found).";
  }

  /**
   * Normalise a Goose message `content` field to a plain string.
   * Goose uses either a string or an array of content blocks:
   *   [{ type: "text", text: "..." }, ...]
   */
  private messageContentToText(content: unknown): string {
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
      return content
        .filter((b: any) => b?.type === "text" && typeof b?.text === "string")
        .map((b: any) => b.text as string)
        .join("\n")
        .trim();
    }
    return "";
  }

  private buildTaskPrompt(task: TaskNode): string {
    const lines = [
      `Task: ${task.title}`,
      `Description: ${task.description}`,
      `Role: ${task.agentRole ?? "generalist"}`,
      `Task ID: ${task.id}`,
      "",
      "Complete this task. Write files, run commands, and provide a final summary.",
    ];

    if (task.dependencies && task.dependencies.length > 0) {
      lines.splice(4, 0, `Dependencies (completed): ${task.dependencies.join(", ")}`);
    }

    return lines.join("\n");
  }
}
