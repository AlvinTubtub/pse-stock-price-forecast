import { NextRequest, NextResponse } from "next/server";
import { GoogleGenAI } from "@google/genai";
import { buildContextForRequest } from "@/lib/ai/context";
import { buildSystemPrompt } from "@/lib/ai/prompt";

const PRIMARY_MODEL = "gemini-3.5-flash-lite";
const FALLBACK_MODEL = "gemini-3.5-flash";
const MAX_MESSAGE_LENGTH = 1000;
const MAX_HISTORY_MESSAGES = 8;
const MAX_WATCHLIST_SYMBOLS = 5;
const PROVIDER_TIMEOUT_MS = 25_000;
const ALLOWED_ROUTES = new Set(["/", "/companies", "/watchlist", "/compare", "/learn", "/learn-stocks", "/about"]);

type ProviderContent = { role: "user" | "model"; parts: Array<{ text: string }> };
type UnknownRecord = Record<string, unknown>;

class EmptyProviderResponseError extends Error {}
class ProviderTimeoutError extends Error {}

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function errorRecord(error: unknown): UnknownRecord {
  return typeof error === "object" && error !== null ? error as UnknownRecord : {};
}

function errorStatus(error: unknown): number | undefined {
  const record = errorRecord(error);
  const value = record.status ?? record.statusCode;
  return typeof value === "number" ? value : undefined;
}

function errorMessage(error: unknown): string {
  const message = errorRecord(error).message;
  return typeof message === "string" ? message.toLowerCase() : "";
}

function isTransientOrAvailabilityError(error: unknown): boolean {
  const status = errorStatus(error);
  const message = errorMessage(error);
  return [502, 503, 504, 404].includes(status ?? 0)
    || ["503", "404", "not_found", "no longer available", "not found", "unavailable", "high demand", "overloaded", "temporarily busy", "service unavailable", "deadline exceeded"]
      .some((token) => message.includes(token));
}

function isRateLimitError(error: unknown): boolean {
  const message = errorMessage(error);
  return errorStatus(error) === 429
    || ["429", "resource_exhausted", "quota"].some((token) => message.includes(token));
}

async function withTimeout<T>(work: Promise<T>, milliseconds: number): Promise<T> {
  let timeoutId: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timeoutId = setTimeout(() => reject(new ProviderTimeoutError("Ask AI provider timeout")), milliseconds);
  });
  try {
    return await Promise.race([work, timeout]);
  } finally {
    if (timeoutId) clearTimeout(timeoutId);
  }
}

async function requestModel(
  ai: GoogleGenAI,
  model: string,
  contents: ProviderContent[],
  systemPrompt: string,
): Promise<string | null> {
  const response = await ai.models.generateContent({
    model,
    contents,
    config: { systemInstruction: systemPrompt, maxOutputTokens: 1000 },
  });
  return response.text?.trim() || null;
}

async function generateWithResilience(
  ai: GoogleGenAI,
  contents: ProviderContent[],
  systemPrompt: string,
): Promise<string> {
  try {
    const reply = await requestModel(ai, PRIMARY_MODEL, contents, systemPrompt);
    if (reply) return reply;
    console.warn("Ask AI primary model returned an empty response; trying fallback.");
  } catch (error: unknown) {
    if (!isTransientOrAvailabilityError(error)) throw error;
    const unavailableModel = errorStatus(error) === 404 || errorMessage(error).includes("not found");
    if (!unavailableModel) {
      console.warn("Ask AI primary model is temporarily unavailable; retrying once.");
      await delay(750);
      try {
        const retryReply = await requestModel(ai, PRIMARY_MODEL, contents, systemPrompt);
        if (retryReply) return retryReply;
      } catch (retryError: unknown) {
        if (!isTransientOrAvailabilityError(retryError)) throw retryError;
      }
    }
  }

  console.warn("Ask AI is using the configured fallback model.");
  const fallbackReply = await requestModel(ai, FALLBACK_MODEL, contents, systemPrompt);
  if (!fallbackReply) throw new EmptyProviderResponseError("Ask AI provider returned no text");
  return fallbackReply;
}

function cleanRoute(value: unknown): string {
  if (typeof value !== "string") return "/";
  const route = value.slice(0, 100);
  if (/^\/companies\/[A-Za-z0-9._-]{1,12}$/.test(route)) return route;
  return ALLOWED_ROUTES.has(route) ? route : "/";
}

function cleanSymbol(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const symbol = value.trim().toUpperCase();
  return /^[A-Z0-9._-]{1,12}$/.test(symbol) ? symbol : undefined;
}

function cleanWatchlist(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  return [...new Set(value.map(cleanSymbol).filter((symbol): symbol is string => Boolean(symbol)))]
    .slice(0, MAX_WATCHLIST_SYMBOLS);
}

function cleanHistory(value: unknown): ProviderContent[] {
  if (!Array.isArray(value)) return [];
  return value.slice(-MAX_HISTORY_MESSAGES).flatMap((item) => {
    if (typeof item !== "object" || item === null) return [];
    const { role, text } = item as UnknownRecord;
    if ((role !== "user" && role !== "assistant") || typeof text !== "string") return [];
    const cleanText = text.trim().slice(0, MAX_MESSAGE_LENGTH);
    return cleanText ? [{ role: role === "user" ? "user" as const : "model" as const, parts: [{ text: cleanText }] }] : [];
  });
}

export async function POST(req: NextRequest) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey?.trim() || apiKey === "your_gemini_api_key_here") {
    console.warn("Ask AI is unavailable because its server-side provider credential is not configured.");
    return NextResponse.json({ error: "Ask AI is temporarily unavailable. Please try again." }, { status: 503 });
  }

  let body: UnknownRecord;
  try {
    const parsed: unknown = await req.json();
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) throw new Error("invalid body");
    body = parsed as UnknownRecord;
  } catch {
    return NextResponse.json({ error: "Unable to read this request. Please try again." }, { status: 400 });
  }

  if (typeof body.message !== "string" || !body.message.trim()) {
    return NextResponse.json({ error: "Please enter a question." }, { status: 400 });
  }
  const message = body.message.trim();
  if (message.length > MAX_MESSAGE_LENGTH) {
    return NextResponse.json({ error: `Please keep your question under ${MAX_MESSAGE_LENGTH} characters.` }, { status: 400 });
  }

  try {
    const route = cleanRoute(body.route);
    const symbol = cleanSymbol(body.symbol);
    const contextData = await buildContextForRequest({ route, symbol, watchlist: cleanWatchlist(body.watchlist) });
    const contents = cleanHistory(body.history);
    contents.push({ role: "user", parts: [{ text: message }] });

    const ai = new GoogleGenAI({ apiKey });
    const reply = await withTimeout(
      generateWithResilience(ai, contents, buildSystemPrompt(contextData)),
      PROVIDER_TIMEOUT_MS,
    );
    return NextResponse.json({ reply });
  } catch (error: unknown) {
    if (isRateLimitError(error)) {
      return NextResponse.json({ error: "Ask AI is temporarily limited. Please try again later." }, { status: 429 });
    }
    if (error instanceof ProviderTimeoutError) {
      return NextResponse.json({ error: "Ask AI took too long to respond. Please try again." }, { status: 504 });
    }
    if (isTransientOrAvailabilityError(error) || error instanceof EmptyProviderResponseError) {
      return NextResponse.json({ error: "Ask AI is temporarily unavailable. Please try again." }, { status: 503 });
    }
    console.error("Ask AI request failed.", { status: errorStatus(error) ?? "unknown" });
    return NextResponse.json({ error: "Ask AI is temporarily unavailable. Please try again." }, { status: 500 });
  }
}
