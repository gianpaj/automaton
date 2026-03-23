/**
 * Goose Worker Pool
 *
 * Spawns Goose (block/goose) CLI subprocesses to execute orchestration tasks.
 * Extends SubprocessWorkerPool — only Goose-specific concerns live here:
 *   - CLI invocation with --provider / --model / --no-session / --output-format json
 *   - JSON output parsing (last assistant text message, with banner-skip)
 *   - Availability check
 *   - Startup cleanup of stale local:// DB entries
 *   - Concurrency cap (maxWorkers)
 *
 * Goose supports Ollama (free, local), Anthropic, OpenAI, Gemini, and many more.
 * With Ollama + qwen2.5:7b this is entirely free and runs offline.
 */

import { SubprocessWorkerPool } from "./subprocess-worker.js";
import type { SubprocessWorkerConfig } from "./subprocess-worker.js";
import type { TaskNode } from "./task-graph.js";

export interface GooseWorkerConfig extends SubprocessWorkerConfig {
  /** LLM provider (e.g. "ollama", "anthropic", "openai"). Omit to use GOOSE_PROVIDER env. */
  provider?: string;
  /** Model ID (e.g. "qwen2.5:7b", "claude-haiku-4-5-20251001"). Omit to use GOOSE_MODEL env. */
  model?: string;
}

export class GooseWorkerPool extends SubprocessWorkerPool {
  constructor(private readonly gooseConfig: GooseWorkerConfig) {
    super(gooseConfig);
    this.markStaleLocalChildrenDead();
    this.checkAvailability();
  }

  /**
   * On startup, mark any local:// children in the DB as 'dead'.
   * Local workers are ephemeral — they only live within the current process.
   * Stale entries from a previous run would otherwise be treated as idle
   * workers by SimpleAgentTracker, causing infinite re-assignment loops.
   */
  private markStaleLocalChildrenDead(): void {
    try {
      const result = this.config.db.prepare(
        `UPDATE children SET status = 'dead'
         WHERE address LIKE 'local://%' AND status IN ('running', 'healthy')`,
      ).run();
      if (result.changes > 0) {
        this.logger.info(`[GOOSE-WORKER] Cleaned up ${result.changes} stale local worker(s) from previous run`);
      }
    } catch (err) {
      this.logger.warn("[GOOSE-WORKER] Failed to clean up stale local workers", {
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }

  /**
   * Override spawn to enforce maxWorkers capacity.
   * Throws synchronously if at capacity so the caller can fall back immediately.
   */
  override spawn(task: TaskNode): { address: string; name: string; sandboxId: string } {
    const maxWorkers = this.config.maxWorkers ?? 2;
    if (this.activeWorkers.size >= maxWorkers) {
      throw new Error(
        `GooseWorkerPool at capacity (${this.activeWorkers.size}/${maxWorkers} workers). ` +
        "Task will be retried when a worker slot is free.",
      );
    }
    return super.spawn(task);
  }

  protected get logTag(): string { return "GOOSE-WORKER"; }

  protected checkAvailability(): void {
    this.execCheck(
      "which goose",
      () => this.logger.warn(
        "[GOOSE-WORKER] Goose not found in PATH. " +
        "Install: curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh | CONFIGURE=false bash",
      ),
      () => {
        const providerInfo = this.gooseConfig.provider
          ? `provider=${this.gooseConfig.provider} model=${this.gooseConfig.model ?? "<goose default>"}`
          : "provider=<GOOSE_PROVIDER env>";
        this.logger.info(`[GOOSE-WORKER] Goose available. ${providerInfo}`);
      },
    );
  }

  protected buildInvocation(prompt: string): { cmd: string; args: string[] } {
    // --no-profile: skip user's goose profile (extensions, settings)
    // --with-builtin developer: re-enable the developer toolbox explicitly
    //   (file write, shell exec, etc.) — needed for tasks to actually act.
    const args = [
      "run", "--no-session", "--no-profile",
      "--with-builtin", "developer",
      "--output-format", "json",
    ];

    if (this.gooseConfig.provider) args.push("--provider", this.gooseConfig.provider);
    if (this.gooseConfig.model) args.push("--model", this.gooseConfig.model);

    args.push("-t", prompt);
    return { cmd: "goose", args };
  }

  /**
   * Parse Goose `--output-format json` output.
   * Goose prints a terminal banner before the JSON, so we skip lines until
   * we find the opening brace, then brace-balance to extract the full JSON object.
   */
  protected parseOutput(stdout: string, workerId: string): string {
    const raw = stdout.trim();
    if (!raw) return "Task completed (no output).";

    const lines = raw.split("\n");
    let jsonStart = -1;
    let braceDepth = 0;

    for (let i = 0; i < lines.length; i++) {
      const trimmed = lines[i].trim();
      if (jsonStart === -1 && trimmed.startsWith("{")) {
        jsonStart = i;
      }
      if (jsonStart !== -1) {
        braceDepth += (trimmed.match(/\{/g) ?? []).length;
        braceDepth -= (trimmed.match(/\}/g) ?? []).length;
        if (braceDepth <= 0) {
          const jsonStr = lines.slice(jsonStart, i + 1).join("\n");
          try {
            const data = JSON.parse(jsonStr) as { messages?: unknown[]; role?: string; content?: unknown };
            if (Array.isArray(data?.messages)) {
              return this.extractLastAssistantText(data.messages, workerId);
            }
            if (data?.role === "assistant") {
              return this.contentToText(data.content);
            }
          } catch { /* malformed JSON — fall through */ }
          break;
        }
      }
    }

    // Bare JSON array fallback
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith("[")) {
        try {
          const arr = JSON.parse(trimmed);
          if (Array.isArray(arr)) {
            return this.extractLastAssistantText(arr, workerId);
          }
        } catch { /* skip */ }
      }
    }

    // Last resort: return the last non-empty, non-banner line
    this.logger.info(`[GOOSE-WORKER ${workerId}] Could not parse JSON output, falling back to plain text`);
    const nonEmpty = lines
      .map((l) => l.trim())
      .filter((l) => l.length > 0 && !l.startsWith("__") && !l.startsWith("\\") && !l.startsWith("L L"));
    return nonEmpty.at(-1) ?? raw;
  }
}
