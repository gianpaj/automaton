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

/** Phrases that indicate the model described steps rather than executing them. */
const HOLLOW_PHRASES = [
  /\bi would\b/i,
  /\byou (should|can|need to|could)\b/i,
  /\bto (accomplish|complete|do) this\b/i,
  /\bhere('s| is) how\b/i,
  /\bfollow(ing)? these steps\b/i,
  /\bthe (following|next) steps?\b/i,
  /\bwould (need to|have to|be)\b/i,
  /\bcan be (done|accomplished|achieved)\b/i,
];

/**
 * Return true if the message list has at least one tool response —
 * meaning the LLM actually called a tool and got a result back.
 */
function hasToolResponses(messages: unknown[]): boolean {
  return messages.some((m) => {
    const msg = m as { role?: string; content?: unknown };
    if (Array.isArray(msg?.content)) {
      return (msg.content as unknown[]).some(
        (block) => (block as any)?.type === "toolResponse",
      );
    }
    return false;
  });
}

/**
 * Return true if the output text looks like a hollow description:
 * contains code blocks but uses hollow phrases suggesting the model
 * explained what to do rather than doing it.
 */
function looksHollow(output: string): boolean {
  const hasCodeBlock = /```/.test(output);
  if (!hasCodeBlock) return false;  // Code blocks without hollow phrases are fine
  return HOLLOW_PHRASES.some((re) => re.test(output));
}

export interface GooseWorkerConfig extends SubprocessWorkerConfig {
  /** LLM provider (e.g. "ollama", "anthropic", "openai"). Omit to use GOOSE_PROVIDER env. */
  provider?: string;
  /** Model ID (e.g. "qwen2.5:7b", "claude-haiku-4-5-20251001"). Omit to use GOOSE_MODEL env. */
  model?: string;
}

export class GooseWorkerPool extends SubprocessWorkerPool {
  /**
   * Messages extracted from the last parseOutput call.
   * Used by validateCompletion to check for actual tool executions.
   * Keyed by workerId so concurrent workers don't interfere.
   */
  private lastMessages = new Map<string, unknown[]>();

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
   * Saves the parsed messages list for use by validateCompletion().
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
              this.lastMessages.set(workerId, data.messages);
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
            this.lastMessages.set(workerId, arr);
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

  /**
   * Detect hollow completions — the model exited 0 but only described steps
   * without actually executing any tools (typical with llama3.2:3b).
   *
   * A completion is hollow when ALL of these are true:
   *   1. No toolResponse messages in the conversation (no tools were called)
   *   2. The output text contains code blocks (looks like instructions)
   *   3. The output contains hollow phrases ("I would", "you should", etc.)
   */
  protected override validateCompletion(_stdout: string, output: string, workerId: string): string | null {
    const messages = this.lastMessages.get(workerId) ?? [];
    this.lastMessages.delete(workerId);  // Clean up

    if (messages.length === 0) return null;  // No messages to check (plain text output)

    if (hasToolResponses(messages)) return null;  // Tools were actually called — genuine completion

    if (looksHollow(output)) {
      return (
        "Worker described the task without executing it — no tool calls were made. " +
        "The model may lack tool-use capability. Consider using qwen2.5:7b or a cloud provider."
      );
    }

    return null;
  }
}
