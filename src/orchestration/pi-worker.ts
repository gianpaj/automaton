/**
 * Pi Worker Pool
 *
 * Spawns Pi (badlogic/pi-mono) CLI subprocesses to execute orchestration tasks.
 * Extends SubprocessWorkerPool — only Pi-specific concerns live here:
 *   - CLI path resolution and invocation
 *   - JSONL output parsing (agent_end event extraction)
 *   - Availability check + credential validation
 *
 * Pi supports Anthropic, OpenAI, Gemini, Groq and other cloud providers.
 */

import path from "node:path";
import fs from "node:fs";
import os from "node:os";
import { fileURLToPath } from "node:url";
import { SubprocessWorkerPool } from "./subprocess-worker.js";
import type { SubprocessWorkerConfig } from "./subprocess-worker.js";
import type { TaskNode } from "./task-graph.js";

/** Env vars that Pi accepts as API credentials (covers all supported providers). */
const PI_CREDENTIAL_ENV_VARS = [
  "ANTHROPIC_API_KEY", "ANTHROPIC_OAUTH_TOKEN",
  "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY",
  "GEMINI_API_KEY", "GROQ_API_KEY",
  "CEREBRAS_API_KEY", "XAI_API_KEY",
  "OPENROUTER_API_KEY", "AI_GATEWAY_API_KEY",
  "ZAI_API_KEY", "MISTRAL_API_KEY",
  "MINIMAX_API_KEY", "OPENCODE_API_KEY",
  "KIMI_API_KEY", "AWS_BEARER_TOKEN_BEDROCK",
  // AWS Bedrock via access key pair
  "AWS_ACCESS_KEY_ID",
];

export interface PiWorkerConfig extends SubprocessWorkerConfig {
  /** Provider for Pi (e.g. "anthropic", "openai"). Omit to use Pi default. */
  provider?: string;
  /** Model for Pi (e.g. "claude-haiku-4-5-20251001", "gpt-4o-mini"). Omit to use Pi default. */
  model?: string;
  /**
   * Override credential detection. Pass `true` to skip the auto-check
   * (useful in tests where env vars and auth files may not be present).
   */
  credentialsVerified?: boolean;
}

export class PiWorkerPool extends SubprocessWorkerPool {
  private readonly cliPath: string;
  /** True once we've confirmed Pi has usable credentials. */
  private credentialsVerified = false;

  constructor(private readonly piConfig: PiWorkerConfig) {
    super(piConfig);
    this.cliPath = findPiCliPath();
    this.credentialsVerified = piConfig.credentialsVerified ?? hasCredentials();
    this.checkAvailability();
  }

  /**
   * Override spawn to throw synchronously if Pi has no LLM credentials.
   * This lets the caller (loop.ts) fall back to Conway sandbox immediately,
   * rather than spawning a subprocess that will fail asynchronously.
   */
  override spawn(task: TaskNode): { address: string; name: string; sandboxId: string } {
    if (!this.credentialsVerified) {
      throw new Error(
        "Pi worker has no LLM credentials. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, " +
        "GEMINI_API_KEY, or another supported API key to use Pi workers.",
      );
    }
    return super.spawn(task);
  }

  protected get logTag(): string { return "PI-WORKER"; }

  protected checkAvailability(): void {
    const cmd = this.cliPath === "pi"
      ? "which pi"
      : `node "${this.cliPath}" --version`;

    this.execCheck(
      cmd,
      () => this.logger.warn(
        "[PI-WORKER] Pi not found. Install with: bun add @mariozechner/pi-coding-agent " +
        "or: npm install -g @mariozechner/pi-coding-agent",
      ),
      () => {
        if (!this.credentialsVerified) {
          this.logger.warn(
            "[PI-WORKER] Pi found but no LLM credentials detected. " +
            "Set ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, or OPENROUTER_API_KEY. " +
            "Pi workers will be skipped and Conway sandbox will be used instead.",
          );
        } else {
          const providerInfo = this.piConfig.provider
            ? `provider=${this.piConfig.provider} model=${this.piConfig.model}`
            : "provider=<pi default>";
          this.logger.info(`[PI-WORKER] Pi available at ${this.cliPath}. ${providerInfo}`);
        }
      },
    );
  }

  protected buildInvocation(prompt: string): { cmd: string; args: string[] } {
    const args: string[] = [];
    let cmd: string;

    if (this.cliPath === "pi") {
      cmd = "pi";
    } else {
      cmd = "node";
      args.push(this.cliPath);
    }

    if (this.piConfig.provider) args.push("--provider", this.piConfig.provider);
    if (this.piConfig.model) args.push("--model", this.piConfig.model);
    args.push("--mode", "json", "--no-session", "-p", prompt);

    return { cmd, args };
  }

  /**
   * Parse Pi `--mode json` JSONL output.
   * Each line is a JSON event. We look for the last `agent_end` event and
   * extract the assistant text from its `messages` array.
   */
  protected parseOutput(stdout: string, workerId: string): string {
    const raw = stdout.trim();
    if (!raw) return "Task completed (no output).";

    // Scan lines in reverse to find the last agent_end event
    for (const line of raw.split("\n").filter(Boolean).reverse()) {
      try {
        const event = JSON.parse(line) as { type?: string; messages?: unknown[] };
        if (event?.type === "agent_end" && Array.isArray(event.messages)) {
          return this.extractLastAssistantText(event.messages, workerId);
        }
      } catch { /* not JSON — skip */ }
    }

    // Fallback: try the whole stdout as a single JSON object
    try {
      const obj = JSON.parse(raw) as { type?: string; messages?: unknown[] };
      if (obj?.type === "agent_end" && Array.isArray(obj.messages)) {
        return this.extractLastAssistantText(obj.messages, workerId);
      }
    } catch { /* not JSON */ }

    // Plain text fallback (older Pi version or --mode json not yet available)
    this.logger.info(`[PI-WORKER ${workerId}] Output is plain text, using as-is`);
    return raw;
  }
}

/** Resolve the Pi CLI path from the installed package, falling back to global binary. */
function findPiCliPath(): string {
  // Strategy 1: ESM import.meta.resolve (Node 20.6+, no flags required)
  try {
    const resolved = import.meta.resolve("@mariozechner/pi-coding-agent");
    // resolved is the package's ESM main (e.g. .../dist/index.js)
    const pkgRoot = path.resolve(path.dirname(fileURLToPath(resolved)), "..");
    const cliPath = path.join(pkgRoot, "dist", "cli.js");
    if (fs.existsSync(cliPath)) return cliPath;
  } catch { /* not available in this Node version */ }

  // Strategy 2: walk up from this file to find node_modules
  try {
    const selfDir = path.dirname(fileURLToPath(import.meta.url));
    // Compiled: dist/orchestration/pi-worker.js  — node_modules is 2 dirs up
    // Source:   src/orchestration/pi-worker.ts   — node_modules is 2 dirs up
    for (let up = 1; up <= 4; up++) {
      const candidate = path.join(selfDir, ...Array(up).fill(".."), "node_modules/@mariozechner/pi-coding-agent/dist/cli.js");
      if (fs.existsSync(candidate)) return path.resolve(candidate);
    }
  } catch { /* ignore */ }

  return "pi";
}

/**
 * Check whether any LLM credentials usable by Pi are available.
 * Looks at environment variables and Pi's ~/.pi/auth.json for stored keys.
 * Keychain-command entries (key starts with "!") are not counted — we cannot
 * verify them without running the keychain command.
 */
function hasCredentials(): boolean {
  // Fast path: any recognised env var is set
  const foundEnvVar = PI_CREDENTIAL_ENV_VARS.find((k) => process.env[k]?.trim());
  if (foundEnvVar) {
    // Log which env var triggered the check (helps diagnose unexpected matches)
    process.stderr.write(`[PI-WORKER hasCredentials] Found env var: ${foundEnvVar}=${process.env[foundEnvVar]?.slice(0, 8)}...\n`);
    return true;
  }

  // Check Pi auth.json for plaintext (non-keychain) API keys
  try {
    const authPath = path.join(os.homedir(), ".pi", "auth.json");
    const raw = fs.readFileSync(authPath, "utf8");
    const auth = JSON.parse(raw) as Record<string, { key?: string }>;
    // Only count entries whose key is a plaintext string, not a shell command (! prefix)
    return Object.values(auth).some(
      (entry) => typeof entry?.key === "string" && !entry.key.startsWith("!") && entry.key.trim().length > 0,
    );
  } catch {
    return false;
  }
}
