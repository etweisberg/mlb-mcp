import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

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

interface PlotData {
  plot_base64: string;
  plot_title: string;
}

interface LLMRequestPayload {
  prompt: string;
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

// Add this helper function for delays
async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Add this retry function with exponential backoff
async function callOpenAIWithRetry(
  requestOptions: OpenAI.Chat.Completions.ChatCompletionCreateParams,
  maxRetries: number = 9,
  baseDelay: number = 1000
): Promise<OpenAI.Chat.Completions.ChatCompletion> {
  let lastError: Error;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await openai.chat.completions.create({
        ...requestOptions,
        stream: false,
      });
      return response as OpenAI.Chat.Completions.ChatCompletion;
    } catch (error) {
      lastError = error as Error;

      // Check if it's a 429 rate limit error
      if (
        error instanceof Error &&
        "status" in error &&
        (error as { status?: number }).status === 429
      ) {
        console.log(
          `Rate limit hit on attempt ${attempt + 1}/${maxRetries + 1}`
        );

        // If this is our last attempt, don't wait
        if (attempt === maxRetries) {
          console.log("Max retries reached, throwing error");
          break;
        }

        // Calculate exponential backoff delay
        const delay = baseDelay * Math.pow(2, attempt);
        const jitter = Math.random() * 0.1 * delay; // Add 10% jitter
        const totalDelay = delay + jitter;

        console.log(
          `Waiting ${Math.round(totalDelay)}ms before retry ${attempt + 2}/${
            maxRetries + 1
          }`
        );
        await sleep(totalDelay);
        continue;
      }

      // For non-429 errors, throw immediately
      throw error;
    }
  }

  throw lastError!;
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

// Parse JSON plot data from LLM response
function parsePlotData(content: string): PlotData[] {
  try {
    // Look for JSON array in the response
    const jsonMatch = content.match(/\[\s*\{[\s\S]*?\}\s*\]/);
    if (jsonMatch) {
      const plotData = JSON.parse(jsonMatch[0]);
      if (Array.isArray(plotData)) {
        // Validate that each item has the required fields
        const validPlots = plotData.filter(
          (item): item is PlotData =>
            typeof item === "object" &&
            item !== null &&
            "plot_base64" in item &&
            "plot_title" in item &&
            typeof item.plot_base64 === "string" &&
            typeof item.plot_title === "string"
        );
        return validPlots;
      }
    }

    // If no JSON found, return empty array
    return [];
  } catch {
    return [];
  }
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
    const result = await mcpClient.callTool({
      name: toolName,
      arguments: parameters,
    });

    // Validate that we actually got data back
    if (!result || !result.content) {
      return JSON.stringify({ error: "Tool returned no data" }, null, 2);
    }

    const resultString = JSON.stringify(result.content, null, 2);
    return resultString;
  } catch (error) {
    // Create a more informative error message but don't lose the original error
    const errorMessage = `Tool execution failed for ${toolName}: ${
      error instanceof Error ? error.message : "Unknown error"
    }`;

    // Re-throw to be handled by the calling function
    throw new Error(errorMessage);
  }
}

function pushMessage(
  messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[],
  nextMsg:
    | OpenAI.Chat.Completions.ChatCompletionMessageParam
    | OpenAI.Chat.Completions.ChatCompletionMessage
): void {
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

  // build the param‐typed message, casting ensures TS compliance
  const msg = {
    role: nextMsg.role as "system" | "user" | "assistant" | "function",
    content: contentStr,
    ...(nextMsg.role === "function" && "name" in nextMsg && nextMsg.name
      ? { name: nextMsg.name }
      : {}),
  } as OpenAI.Chat.Completions.ChatCompletionMessageParam;

  messages.push(msg);
}

async function callOpenAI(payload: LLMRequestPayload): Promise<PlotData[]> {
  let mcpClient: Client | null = null;
  try {
    const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
      { role: "user", content: payload.prompt },
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

      // Check if this is the final iteration
      const isFinalIteration = iteration >= maxIterations;

      if (isFinalIteration) {
        // Add a final instruction for the last iteration
        messages.push({
          role: "user",
          content:
            "This is the final response. Please provide your plot data in the exact JSON format specified.",
        });
      }

      const response = await callOpenAIWithRetry({
        ...requestOptions,
        messages,
      });

      const message = response.choices[0]?.message;
      if (!message) {
        throw new Error("No response from OpenAI API");
      }

      // Add message to conversation
      pushMessage(messages, message);

      // Check if there are tool calls to execute (but not on final iteration)
      if (
        message.tool_calls &&
        message.tool_calls.length > 0 &&
        mcpClient &&
        !isFinalIteration
      ) {
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

            pushMessage(messages, toolMsg);
          } catch (toolError) {
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
            pushMessage(messages, errorToolMsg);
          }
        }

        // Continue the loop to get the next response
        continue;
      } else {
        // No tool calls or final iteration, we have the final response
        const plotData = parsePlotData(message.content || "");
        return plotData;
      }
    }

    // If we've hit the max iterations, try to parse the last message
    const lastMessage = messages[messages.length - 1];
    if (
      lastMessage &&
      lastMessage.role === "assistant" &&
      "content" in lastMessage
    ) {
      const content =
        typeof lastMessage.content === "string" ? lastMessage.content : "";
      return parsePlotData(content);
    }

    throw new Error("Max iterations reached without final response");
  } catch (error) {
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

    if (!process.env.OPENAI_API_KEY) {
      return NextResponse.json(
        { error: "OpenAI API key not configured" },
        { status: 500 }
      );
    }

    const plotData = await callOpenAI(payload);

    if (plotData.length === 0) {
      return NextResponse.json(
        {
          error: "No plot data found in LLM response",
        },
        { status: 422 }
      );
    }

    // Return the plot data as JSON
    return NextResponse.json(plotData);
  } catch (error) {
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
