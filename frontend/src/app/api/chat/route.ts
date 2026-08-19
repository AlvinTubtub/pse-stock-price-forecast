import { NextRequest, NextResponse } from "next/server";
import { GoogleGenAI } from "@google/genai";
import { buildContextForRequest } from "@/lib/ai/context";
import { buildSystemPrompt } from "@/lib/ai/prompt";

const PRIMARY_MODEL = "gemini-3.5-flash-lite";
const FALLBACK_MODEL = "gemini-3.5-flash";
const MAX_MESSAGE_LENGTH = 1000;
const MAX_HISTORY_TURNS = 6;

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function isTransientOrAvailabilityError(error: any): boolean {
  const status = error?.status || error?.statusCode;
  const msg = String(error?.message || "").toLowerCase();

  return (
    status === 503 ||
    status === 502 ||
    status === 504 ||
    status === 404 ||
    msg.includes("503") ||
    msg.includes("404") ||
    msg.includes("not_found") ||
    msg.includes("no longer available") ||
    msg.includes("not found") ||
    msg.includes("unavailable") ||
    msg.includes("high demand") ||
    msg.includes("overloaded") ||
    msg.includes("temporarily busy") ||
    msg.includes("service unavailable") ||
    msg.includes("deadline exceeded")
  );
}

function isRateLimitError(error: any): boolean {
  const status = error?.status || error?.statusCode;
  const msg = String(error?.message || "").toLowerCase();

  return (
    status === 429 ||
    msg.includes("429") ||
    msg.includes("resource_exhausted") ||
    msg.includes("quota")
  );
}

async function generateWithResilience(
  ai: GoogleGenAI,
  contents: Array<{ role: "user" | "model"; parts: Array<{ text: string }> }>,
  systemPrompt: string
): Promise<string> {
  const config = {
    systemInstruction: systemPrompt,
    maxOutputTokens: 1000,
  };

  let primaryFailed = false;

  // Attempt 1: Primary Model (gemini-3.5-flash-lite)
  try {
    const response = await ai.models.generateContent({
      model: PRIMARY_MODEL,
      contents,
      config,
    });
    const text = response.text?.trim();
    if (text) return text;
  } catch (error: any) {
    if (!isTransientOrAvailabilityError(error)) {
      throw error;
    }
    primaryFailed = true;
    const msg = String(error?.message || "").toLowerCase();
    const isNotFound =
      error?.status === 404 ||
      msg.includes("404") ||
      msg.includes("not found") ||
      msg.includes("no longer available");

    if (isNotFound) {
      console.warn(
        `Gemini primary model (${PRIMARY_MODEL}) not available or deprecated. Immediately switching to fallback: ${FALLBACK_MODEL}`
      );
    } else {
      console.warn(
        `Gemini primary model unavailable: ${PRIMARY_MODEL}. Retrying primary model after short delay...`
      );

      // Attempt 2: Retry Primary Model once after short delay (only for transient errors, not 404s)
      await delay(750);
      try {
        const response = await ai.models.generateContent({
          model: PRIMARY_MODEL,
          contents,
          config,
        });
        const text = response.text?.trim();
        if (text) return text;
        primaryFailed = false;
      } catch (retryError: any) {
        if (!isTransientOrAvailabilityError(retryError)) {
          throw retryError;
        }
        console.warn(
          `Gemini primary model (${PRIMARY_MODEL}) failed on retry. Attempting fallback model: ${FALLBACK_MODEL}`
        );
      }
    }
  }

  // Attempt 3: Fallback Model (gemini-3.5-flash)
  if (primaryFailed) {
    try {
      const response = await ai.models.generateContent({
        model: FALLBACK_MODEL,
        contents,
        config,
      });
      const text = response.text?.trim();
      if (text) return text;
    } catch (error: any) {
      console.error(
        `Gemini fallback model (${FALLBACK_MODEL}) also failed:`,
        error?.message || "Unknown error"
      );
      throw error;
    }
  }

  return "I was unable to generate a response. Please try again.";
}

export async function POST(req: NextRequest) {
  try {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey || apiKey.trim() === "" || apiKey === "your_gemini_api_key_here") {
      return NextResponse.json(
        {
          error:
            "AI assistant is currently unavailable. GEMINI_API_KEY is not configured in environment.",
        },
        { status: 503 }
      );
    }

    let body: any;
    try {
      body = await req.json();
    } catch {
      return NextResponse.json(
        { error: "Invalid JSON request body." },
        { status: 400 }
      );
    }

    const { message, route, symbol, history } = body || {};

    // 1. Validate message
    if (!message || typeof message !== "string" || message.trim().length === 0) {
      return NextResponse.json(
        { error: "Please provide a valid question or message." },
        { status: 400 }
      );
    }

    const cleanMessage = message.trim();
    if (cleanMessage.length > MAX_MESSAGE_LENGTH) {
      return NextResponse.json(
        {
          error: `Message is too long. Please keep your question under ${MAX_MESSAGE_LENGTH} characters.`,
        },
        { status: 400 }
      );
    }

    // 2. Build page-aware context & system prompt
    const contextData = await buildContextForRequest({ route, symbol });
    const systemPrompt = buildSystemPrompt(contextData);

    // 3. Format conversational contents for Gemini API
    const contents: Array<{ role: "user" | "model"; parts: Array<{ text: string }> }> = [];

    if (Array.isArray(history)) {
      const recentHistory = history.slice(-MAX_HISTORY_TURNS);
      for (const item of recentHistory) {
        if (item && typeof item.text === "string" && item.text.trim()) {
          const role: "user" | "model" = item.role === "user" ? "user" : "model";
          contents.push({
            role,
            parts: [{ text: item.text.trim() }],
          });
        }
      }
    }

    // Add current user prompt
    contents.push({
      role: "user",
      parts: [{ text: cleanMessage }],
    });

    // 4. Invoke Gemini API with primary, retry, and fallback resilience
    const ai = new GoogleGenAI({ apiKey });
    const replyText = await generateWithResilience(ai, contents, systemPrompt);

    return NextResponse.json({ reply: replyText });
  } catch (error: any) {
    const errMsg = String(error?.message || "");

    // Handle 429 / Quota / Rate Limits
    if (isRateLimitError(error)) {
      return NextResponse.json(
        { error: "AI usage is temporarily limited. Please try again later." },
        { status: 429 }
      );
    }

    // Handle 503 / 404 / Transient availability failures
    if (isTransientOrAvailabilityError(error)) {
      return NextResponse.json(
        { error: "Gemini is temporarily busy. Please try again in a moment." },
        { status: 503 }
      );
    }

    // Prevent exposing internal errors or keys
    console.error("Gemini API Error:", errMsg);
    return NextResponse.json(
      { error: "Unable to process request at this time. Please try again later." },
      { status: 500 }
    );
  }
}
