import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import {
  getEncoding,
  encodingForModel,
  type TiktokenModel,
  type Tiktoken,
} from "js-tiktoken";

interface MCPTool {
  name: string;
  description?: string;
  inputSchema: {
    type: "object";
    properties?: Record<string, unknown>;
    required?: string[];
  };
  outputSchema?: Record<string, unknown>;
  annotations?: Record<string, unknown>;
}

interface LLMRequestPayload {
  prompt: string;
  systemPrompt: string;
  availableTools: MCPTool[];
  modelConfig: {
    model: string;
    maxTokens: number;
  };
}

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Model configuration with rate limits (tokens per minute)
const OPENAI_MODELS = {
  "gpt-4o-mini": 120_000,
  "gpt-4.1-nano": 200_000,
  "gpt-4.1-mini": 200_000,
} as const;

type OpenAIModel = keyof typeof OPENAI_MODELS;

// Rate limiting state
interface RateLimitState {
  tokensUsed: number;
  windowStart: number;
  requestCount: number;
}

// In-memory rate limit tracking (in production, use Redis or similar)
const rateLimitState = new Map<OpenAIModel, RateLimitState>();

// Cache for js-tiktoken encoders to avoid repeated initialization
const encoderCache = new Map<string, Tiktoken>();

// Initialize rate limit state for all models
Object.keys(OPENAI_MODELS).forEach((model) => {
  rateLimitState.set(model as OpenAIModel, {
    tokensUsed: 0,
    windowStart: Date.now(),
    requestCount: 0,
  });
});

// Helper function to validate OpenAI model
function validateOpenAIModel(model: string): model is OpenAIModel {
  return model in OPENAI_MODELS;
}

// Get or create js-tiktoken encoder for a model
function getEncoder(model: string): Tiktoken {
  if (encoderCache.has(model)) {
    const cachedEncoder = encoderCache.get(model);
    if (cachedEncoder) {
      return cachedEncoder;
    }
  }

  let encoder: Tiktoken;
  const encodingName = "o200k_base";
  try {
    // Try to get encoding specifically for the model if it's a valid js-tiktoken model
    encoder = encodingForModel(model as TiktokenModel);
    console.log(`Encoder created: ${encoder}`);
  } catch (error) {
    console.warn(
      `Using fallback encoding ${encodingName} for model ${model} - ${error}`
    );
    encoder = getEncoding(encodingName);
  }

  encoderCache.set(model, encoder);
  return encoder;
}

// Accurate token count using js-tiktoken
function countTokens(text: string, model: string): number {
  try {
    const encoder = getEncoder(model);
    const tokens = encoder.encode(text);
    return tokens.length;
  } catch (error) {
    console.error(`Error counting tokens for model ${model}:`, error);
    // Fallback to rough estimation
    return Math.ceil(text.length / 4);
  }
}

// Count tokens for messages array (includes message formatting overhead)
function countMessagesTokens(
  messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[],
  model: string
): number {
  const encoder = getEncoder(model);
  let totalTokens = 0;

  for (const message of messages) {
    // Account for message formatting tokens
    totalTokens += 4; // Base tokens per message (role, content wrapper, etc.)

    if (message.role) {
      totalTokens += encoder.encode(message.role).length;
    }

    if ("content" in message && typeof message.content === "string") {
      totalTokens += encoder.encode(message.content).length;
    }

    if ("name" in message && message.name) {
      totalTokens += encoder.encode(message.name).length;
      totalTokens += 1; // Additional token for name field
    }
  }

  // Add tokens for function calling overhead if tools are present
  totalTokens += 2; // Priming tokens for assistant response

  return totalTokens;
}

// Count tokens for tools (function definitions)
function countToolsTokens(tools: MCPTool[], model: string): number {
  if (tools.length === 0) return 0;

  const encoder = getEncoder(model);
  let totalTokens = 0;

  for (const tool of tools) {
    // Serialize the tool definition as it would appear to the model
    const toolDefinition = JSON.stringify({
      type: "function",
      function: {
        name: tool.name,
        description: tool.description || `Execute ${tool.name}`,
        parameters: tool.inputSchema || {},
      },
    });

    totalTokens += encoder.encode(toolDefinition).length;
  }

  // Add overhead for tools structure
  totalTokens += tools.length * 3; // Approximate overhead per tool

  return totalTokens;
}

// Check and update rate limit
function checkRateLimit(
  model: OpenAIModel,
  estimatedTokens: number
): { allowed: boolean; remaining: number } {
  const state = rateLimitState.get(model)!;
  const now = Date.now();
  const windowDuration = 60 * 1000; // 1 minute
  const maxTokens = OPENAI_MODELS[model];

  // Reset window if it's been more than a minute
  if (now - state.windowStart >= windowDuration) {
    state.tokensUsed = 0;
    state.windowStart = now;
    state.requestCount = 0;
  }

  const wouldExceed = state.tokensUsed + estimatedTokens > maxTokens;
  const remaining = maxTokens - state.tokensUsed;

  if (!wouldExceed) {
    state.tokensUsed += estimatedTokens;
    state.requestCount++;
  }

  return { allowed: !wouldExceed, remaining };
}

// Truncate prompt to fit within rate limits using js-tiktoken
function truncatePromptForRateLimit(
  prompt: string,
  maxTokens: number,
  model: string
): { prompt: string; truncated: boolean } {
  const currentTokens = countTokens(prompt, model);
  console.log(`Current tokens: ${currentTokens} | Max tokens: ${maxTokens}`);

  if (currentTokens <= maxTokens) {
    return { prompt, truncated: false };
  }

  // Reserve tokens for the truncation message
  const truncationMessage =
    "\n\n[PROMPT TRUNCATED DUE TO RATE LIMITS - This is your final prompt, please provide your best response based on the available information.]";
  const encoder = getEncoder(model);
  const truncationMessageTokens = encoder.encode(truncationMessage);
  const promptTokens = encoder.encode(prompt);
  const truncatedTokens = [...truncationMessageTokens, ...promptTokens].slice(
    0,
    maxTokens
  );
  const truncatedPrompt = encoder.decode(truncatedTokens);
  return { prompt: truncatedPrompt, truncated: true };
}

// Convert tools to OpenAI format
function convertToOpenAITools(tools: MCPTool[]) {
  return tools.map((tool) => ({
    type: "function" as const,
    function: {
      name: tool.name,
      description: tool.description || `Execute ${tool.name}`,
      parameters: tool.inputSchema || {},
    },
  }));
}

// Extract HTML from response
function extractHTML(content: string): string {
  // drop everything before the first "<html" or "<!DOCTYPE"
  const startIdx = content.search(/<!DOCTYPE html|<html/i);
  if (startIdx >= 0) {
    content = content.slice(startIdx);
  }

  // try code‐fence first
  const fenceRe = /```(?:html)?\s*([\s\S]*?)\s*```/i;
  const fenceMatch = content.match(fenceRe);
  if (fenceMatch && fenceMatch[1]) {
    return fenceMatch[1].trim();
  }

  // fallback to a bare HTML match
  const htmlRe = /<!DOCTYPE html[\s\S]*?<\/html>|<html[\s\S]*?<\/html>/i;
  const htmlMatch = content.match(htmlRe);
  if (htmlMatch) {
    return htmlMatch[0].trim();
  }

  return "";
}

// MCP client instance
let globalMCPClient: Client | null = null;
let clientInitializationPromise: Promise<Client> | null = null;

// Initialize MCP client with connection
async function getMCPClient(): Promise<Client> {
  // If client already exists and is connected, return it
  if (globalMCPClient) {
    return globalMCPClient;
  }

  // If initialization is already in progress, wait for it
  if (clientInitializationPromise) {
    return clientInitializationPromise;
  }

  // Start new initialization
  clientInitializationPromise = (async () => {
    const url = process.env.NEXT_PUBLIC_MLB_STATS_MCP_URL;
    if (!url) {
      throw new Error(
        "NEXT_PUBLIC_MLB_STATS_MCP_URL environment variable is not set"
      );
    }

    const client = new Client({
      name: "ai-baseball-analyst-backend",
      version: "1.0.0",
    });

    const transport = new StreamableHTTPClientTransport(new URL(url));
    await client.connect(transport);

    globalMCPClient = client;
    return client;
  })();

  try {
    return await clientInitializationPromise;
  } catch (error) {
    // Reset on error so we can retry
    clientInitializationPromise = null;
    globalMCPClient = null;
    throw error;
  }
}

// Execute an MCP Tool Call
async function executeToolCall(
  toolName: string,
  parameters: Record<string, unknown>,
  mcpClient: Client
): Promise<string> {
  try {
    console.log(`Executing tool: ${toolName} with parameters:`, parameters);

    const result = await mcpClient.callTool({
      name: toolName,
      arguments: parameters,
    });

    // Validate that we actually got data back
    if (!result || !result.content) {
      console.warn(`Tool ${toolName} returned empty or null result`);
      return JSON.stringify({ error: "Tool returned no data" }, null, 2);
    }

    // Check for successful data retrieval
    const hasValidContent = Array.isArray(result.content)
      ? result.content.length > 0
      : result.content && typeof result.content === "object";

    if (!hasValidContent) {
      console.warn(
        `Tool ${toolName} returned empty content array or invalid data structure`
      );
    }
    const resultString = JSON.stringify(result.content, null, 2);
    console.log(
      `Tool ${toolName} executed successfully with ${
        hasValidContent ? "valid" : "empty"
      } content`
    );
    return resultString;
  } catch (error) {
    console.error(`Error executing tool ${toolName}:`, error);

    // Create a more informative error message but don't lose the original error
    const errorMessage = `Tool execution failed for ${toolName}: ${
      error instanceof Error ? error.message : "Unknown error"
    }`;

    // Re-throw to be handled by the calling function
    throw new Error(errorMessage);
  }
}

function safePushMessage(
  messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[],
  nextMsg:
    | OpenAI.Chat.Completions.ChatCompletionMessageParam
    | OpenAI.Chat.Completions.ChatCompletionMessage,
  model: OpenAIModel
): boolean {
  // flatten content
  const raw = nextMsg.content ?? "";
  const contentStr =
    typeof raw === "string"
      ? raw
      : Array.isArray(raw)
      ? raw
          .map((p) => ("text" in p && typeof p.text === "string" ? p.text : ""))
          .join("")
      : "";

  // check budget
  const needed = countTokens(contentStr, model);
  const { allowed, remaining } = checkRateLimit(model, needed);

  // decide what to push
  const toPush = allowed
    ? contentStr
    : (() => {
        const { prompt, truncated } = truncatePromptForRateLimit(
          contentStr,
          remaining,
          model
        );
        if (!truncated) throw new Error("safePushMessage: truncate failed");
        return prompt;
      })();

  // build the param‐typed message, casting ensures TS compliance
  const msg = {
    role: nextMsg.role as "system" | "user" | "assistant" | "function",
    content: toPush,
    ...(nextMsg.role === "function" && "name" in nextMsg && nextMsg.name
      ? { name: nextMsg.name }
      : {}),
  } as OpenAI.Chat.Completions.ChatCompletionMessageParam;

  messages.push(msg);
  return allowed;
}

async function callOpenAI(payload: LLMRequestPayload): Promise<string> {
  let mcpClient: Client | null = null;
  try {
    // Validate model
    if (!validateOpenAIModel(payload.modelConfig.model)) {
      throw new Error(`Unsupported model: ${payload.modelConfig.model}`);
    }

    const model = payload.modelConfig.model as OpenAIModel;

    // Build initial messages
    const initialMessages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] =
      [
        { role: "system", content: payload.systemPrompt },
        { role: "user", content: payload.prompt },
      ];

    // Calculate accurate token estimates
    const messagesTokens = countMessagesTokens(initialMessages, model);
    const toolsTokens = countToolsTokens(payload.availableTools, model);
    const estimatedTokens = messagesTokens + toolsTokens;

    // Check rate limit
    const rateLimitCheck = checkRateLimit(model, estimatedTokens);
    let currentPrompt = payload.prompt;
    let isRateLimited = false;

    if (!rateLimitCheck.allowed) {
      console.warn(
        `Rate limit would be exceeded. Remaining tokens: ${rateLimitCheck.remaining}`
      );

      // Calculate how many tokens we can use for the prompt
      const systemTokens = countTokens(payload.systemPrompt, model);
      const availableForPrompt =
        rateLimitCheck.remaining - systemTokens - toolsTokens - 1000; // Buffer

      const truncateResult = truncatePromptForRateLimit(
        payload.prompt,
        availableForPrompt,
        model
      );
      currentPrompt = truncateResult.prompt;
      isRateLimited = true;
    }

    const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
      { role: "system", content: payload.systemPrompt },
      { role: "user", content: currentPrompt },
    ];

    const requestOptions: OpenAI.Chat.Completions.ChatCompletionCreateParams = {
      model: payload.modelConfig.model,
      messages,
      max_tokens: payload.modelConfig.maxTokens,
    };

    if (payload.availableTools.length > 0) {
      requestOptions.tools = convertToOpenAITools(payload.availableTools);
      requestOptions.tool_choice = "auto";
    }

    // Get MCP client for tool execution
    if (payload.availableTools.length > 0) {
      try {
        mcpClient = await getMCPClient();
      } catch (mcpError) {
        console.error("Failed to get MCP client:", mcpError);
        throw new Error(
          `MCP client error: ${
            mcpError instanceof Error ? mcpError.message : "Unknown error"
          }`
        );
      }
    }

    // Conversational loop - continue until no more tool calls
    const maxIterations = 10; // Prevent infinite loops
    let iteration = 0;

    while (iteration < maxIterations) {
      iteration++;
      // Log token usage before each request
      const currentTokenCount = countMessagesTokens(messages, model);
      console.log(`Iteration ${iteration} token count: ${currentTokenCount}`);

      // Check if this is the final iteration due to rate limits or max iterations
      const isFinalIteration = isRateLimited || iteration >= maxIterations;

      if (isFinalIteration && !isRateLimited) {
        // Add a final instruction for the last iteration if not rate limited
        messages.push({
          role: "user",
          content:
            "This is the final response. Please provide your best final HTML based on all the information gathered so far.",
        });
      }

      const response = await openai.chat.completions.create({
        ...requestOptions,
        messages,
      });

      const message = response.choices[0]?.message;
      if (!message) {
        throw new Error("No response from OpenAI API");
      }

      // Get actual token usage
      if (response.usage) {
        console.log(`Actual token usage:`, response.usage);
        const tokensUsed = response.usage.total_tokens;
        const state = rateLimitState.get(model)!;
        state.tokensUsed = tokensUsed;
      }

      // Perform safe push of message
      if (!safePushMessage(messages, message, model)) {
        break;
      }

      // Check if there are tool calls to execute (but not on final iteration)
      if (
        message.tool_calls &&
        message.tool_calls.length > 0 &&
        mcpClient &&
        !isFinalIteration
      ) {
        console.log(
          `Processing ${message.tool_calls.length} tool calls in iteration ${iteration}...`
        );

        for (const toolCall of message.tool_calls) {
          try {
            const result = await executeToolCall(
              toolCall.function.name,
              JSON.parse(toolCall.function.arguments),
              mcpClient
            );

            const toolMsg: OpenAI.Chat.Completions.ChatCompletionMessageParam =
              {
                role: "function",
                name: toolCall.function.name,
                content: result,
              };

            if (!safePushMessage(messages, toolMsg, model)) {
              break;
            }
          } catch (toolError) {
            console.error(
              `Tool execution failed for ${toolCall.function.name}:`,
              toolError
            );
            // Add error result to conversation so the LLM can handle it
            const errorToolMsg: OpenAI.Chat.Completions.ChatCompletionMessageParam =
              {
                role: "function",
                content: `Error executing ${toolCall.function.name}: ${
                  toolError instanceof Error
                    ? toolError.message
                    : "Unknown error"
                }`,
                name: toolCall.function.name,
              };
            if (!safePushMessage(messages, errorToolMsg, model)) {
              break;
            }
          }
        }

        // Continue the loop to get the next response
        continue;
      } else {
        // No tool calls or final iteration, we have the final response
        console.log(
          `OpenAI conversation completed after ${iteration} iterations`
        );
        return message.content || "";
      }
    }

    // If we've hit the max iterations, return the last message content
    const lastMessage = messages[messages.length - 1];
    if (
      lastMessage &&
      lastMessage.role === "assistant" &&
      "content" in lastMessage
    ) {
      console.warn(`Returning last response: ${lastMessage.content}`);
      return typeof lastMessage.content === "string" ? lastMessage.content : "";
    }

    throw new Error("Max iterations reached without final response");
  } catch (error) {
    console.error("Error in callOpenAI:", error);
    throw error;
  }
}

export async function POST(request: NextRequest) {
  try {
    const payload: LLMRequestPayload = await request.json();

    // Validate required fields
    if (!payload.prompt || !payload.modelConfig?.model) {
      return NextResponse.json(
        { error: "Missing required fields: prompt and modelConfig.model" },
        { status: 400 }
      );
    }

    // Validate that it's a supported OpenAI model
    if (!validateOpenAIModel(payload.modelConfig.model)) {
      return NextResponse.json(
        {
          error: `Unsupported model: ${
            payload.modelConfig.model
          }. Supported models: ${Object.keys(OPENAI_MODELS).join(", ")}`,
        },
        { status: 400 }
      );
    }

    if (!process.env.OPENAI_API_KEY) {
      return NextResponse.json(
        { error: "OpenAI API key not configured" },
        { status: 500 }
      );
    }

    const llmResponse = await callOpenAI(payload);

    // Extract HTML from the response
    const htmlResponse = extractHTML(llmResponse);

    console.log(
      `LLM Response: ${llmResponse} | Extracted HTML: ${htmlResponse}`
    );

    if (!htmlResponse) {
      return NextResponse.json(
        {
          error: "No HTML content found in LLM response",
          fullResponse: llmResponse,
        },
        { status: 422 }
      );
    }

    // Return only the HTML content
    return new NextResponse(htmlResponse, {
      headers: {
        "Content-Type": "text/html",
      },
    });
  } catch (error) {
    console.error("LLM API Error:", error);

    // Handle rate limit errors and other API-specific errors
    if (error instanceof Error) {
      if (error.message.includes("429")) {
        return NextResponse.json(
          { error: `Rate limit: ${error.message}`, type: "rate_limit" },
          { status: 429 }
        );
      }
      if (error.message.includes("401")) {
        return NextResponse.json(
          { error: "Invalid API key", type: "auth_error" },
          { status: 401 }
        );
      }
    }

    return NextResponse.json(
      {
        error: "Internal server error",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
