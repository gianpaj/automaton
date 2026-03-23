/**
 * Ollama Model Auto-Detection
 *
 * Queries the local Ollama server for available models and selects the best
 * one for tool-using agentic tasks. Prefers models known to execute tool calls
 * (qwen2.5, qwen2.5-coder, llama3.1+) over ones that tend to only describe
 * tool calls in text (llama3.2:3b, smaller llama variants).
 *
 * Priority tiers (higher = better tool use):
 *   3 — qwen2.5 7b+ / qwen2.5-coder (excellent tool use)
 *   2 — llama3.1 8b+ / qwen3 4b+ / mistral (good tool use)
 *   1 — any other model (unknown capability)
 *   0 — models known to be poor at tool use (llama3.2:1b/3b)
 */

import { createLogger } from "../observability/logger.js";

const logger = createLogger("orchestration.ollama-detect");

interface OllamaModel {
  name: string;
  size?: number;
}

interface OllamaTagsResponse {
  models: OllamaModel[];
}

/** Models known to poorly execute tool calls (describe instead of acting). */
const POOR_TOOL_USE_PATTERNS = [
  /^llama3\.2:1b/i,
  /^llama3\.2:3b/i,
  /^llama3:(?!.*[89]b|.*\d{2,}b)/i,   // llama3 without 8b/70b suffix
];

/** Score a model name for tool-use capability. Higher is better. */
function scoreModel(name: string): number {
  const lower = name.toLowerCase();

  // Tier 0: explicitly poor
  if (POOR_TOOL_USE_PATTERNS.some((re) => re.test(name))) return 0;

  // Tier 3: best known tool use
  if (/qwen2\.5/.test(lower)) return 3;
  if (/qwen2\.5-coder/.test(lower)) return 3;

  // Tier 2: good tool use
  if (/qwen3/.test(lower)) return 2;
  if (/llama3\.1/.test(lower)) return 2;
  if (/llama3\.3/.test(lower)) return 2;
  if (/mistral/.test(lower)) return 2;
  if (/mixtral/.test(lower)) return 2;
  if (/deepseek/.test(lower)) return 2;
  if (/gemma2/.test(lower)) return 2;

  // Tier 1: unknown capability
  return 1;
}

/**
 * Query Ollama and return the best available model for tool-using tasks.
 * Returns null if Ollama is unreachable or has no models.
 */
export async function detectBestOllamaModel(baseUrl = "http://localhost:11434"): Promise<string | null> {
  try {
    const res = await fetch(`${baseUrl}/api/tags`, {
      signal: AbortSignal.timeout(3000),
    });

    if (!res.ok) {
      logger.warn(`[ollama-detect] /api/tags returned ${res.status}`);
      return null;
    }

    const data = (await res.json()) as OllamaTagsResponse;
    const models = data?.models ?? [];

    if (models.length === 0) {
      logger.info("[ollama-detect] Ollama has no models installed");
      return null;
    }

    // Score and sort — stable sort preserves original order for equal scores
    const scored = models
      .map((m) => ({ name: m.name, score: scoreModel(m.name) }))
      .sort((a, b) => b.score - a.score);

    const best = scored[0];
    const modelList = scored.map((m) => `${m.name}(${m.score})`).join(", ");
    logger.info(`[ollama-detect] Available models: ${modelList}`);
    logger.info(`[ollama-detect] Selected: ${best.name} (score=${best.score})`);

    return best.name;
  } catch (err) {
    // Ollama not running or network error — not an error condition, just no Ollama
    logger.info("[ollama-detect] Ollama not reachable, skipping auto-detect", {
      error: err instanceof Error ? err.message : String(err),
    });
    return null;
  }
}
