"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useMCP } from "@/lib/mcp-context";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Loader2, ArrowLeft, RefreshCw } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";

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

export default function ResultsPage() {
  const router = useRouter();
  const [plotResults, setPlotResults] = useState<PlotData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [requestPayload, setRequestPayload] =
    useState<LLMRequestPayload | null>(null);
  const [hasGenerated, setHasGenerated] = useState(false);
  const [submissionData, setSubmissionData] = useState<{
    completePromptText: string;
    model?: string;
  } | null>(null);
  const { client: mcpClient, tools } = useMCP();

  useEffect(() => {
    // Get submission data from sessionStorage
    const storedData = sessionStorage.getItem("baseball-submission-data");
    if (!storedData) {
      router.push("/"); // Redirect to home if no data
      return;
    }

    try {
      const data = JSON.parse(storedData);
      setSubmissionData(data);
    } catch (error) {
      console.error("Error parsing submission data:", error);
      setError("Invalid submission data");
      setLoading(false);
    }
  }, [router]);

  const prepareRequestPayload = useCallback(
    async (prompt: string, model?: string): Promise<LLMRequestPayload> => {
      const availableTools = tools;

      return {
        prompt,
        availableTools: availableTools,
        modelConfig: {
          model: model || "gpt-4.1-nano",
          maxTokens: 8000,
        },
      };
    },
    [tools]
  );

  const generateReport = useCallback(
    async (prompt: string, model?: string) => {
      try {
        setLoading(true);
        setError(null);

        const payload = await prepareRequestPayload(prompt, model);
        setRequestPayload(payload);

        console.log("Request payload prepared for server-side API:");
        console.log(payload);

        const response = await fetch("/api/llm", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        });

        console.log(response);

        if (!response.ok) {
          const errorData = await response.json();
          console.error(errorData);
          throw new Error(
            `API error: ${errorData.error || response.statusText}`
          );
        }

        // The API returns JSON with plot data
        const plotData: PlotData[] = await response.json();
        setPlotResults(plotData);
      } catch (err) {
        console.error("Error generating report:", err);
        setError(
          `Failed to generate report: ${
            err instanceof Error ? err.message : "Unknown error"
          }`
        );
      } finally {
        setLoading(false);
      }
    },
    [prepareRequestPayload]
  );

  const initializeAndGenerate = useCallback(
    async (data: { completePromptText: string; model?: string }) => {
      if (hasGenerated) {
        return;
      }
      setHasGenerated(true);
      if (!mcpClient) {
        setError("MCP client not available");
        setLoading(false);
        return;
      }

      try {
        // Generate the report using the shared client
        await generateReport(data.completePromptText, data.model);
      } catch (error) {
        console.error("Error generating report:", error);
        setError(
          `Failed to generate report: ${
            error instanceof Error ? error.message : "Unknown error"
          }`
        );
        setLoading(false);
      }
    },
    [mcpClient, hasGenerated, generateReport]
  );

  // Effect to generate report when both submission data and MCP client are ready
  useEffect(() => {
    if (submissionData && mcpClient && tools.length > 0) {
      initializeAndGenerate(submissionData);
    }
  }, [submissionData, mcpClient, tools, initializeAndGenerate]);

  const handleRetry = () => {
    if (submissionData && mcpClient) {
      generateReport(submissionData.completePromptText, submissionData.model);
    }
  };

  const handleBackToHome = () => {
    // Clear the submission data and navigate back
    sessionStorage.removeItem("baseball-submission-data");
    router.push("/");
  };

  return (
    <div className="dark min-h-screen bg-background text-foreground py-8">
      <div className="max-w-6xl mx-auto px-4">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Button
              variant="outline"
              size="sm"
              onClick={handleBackToHome}
              className="flex items-center gap-2"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to Home
            </Button>
            <div>
              <h1 className="text-4xl font-bold text-foreground">
                Baseball Analysis 📊
              </h1>
              <p className="text-muted-foreground">
                AI-generated visualizations using MCP Connected to MLB data and
                advanced analytics
              </p>
            </div>
          </div>
          {!loading && !error && plotResults.length > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleRetry}
              className="flex items-center gap-2"
            >
              <RefreshCw className="h-4 w-4" />
              Regenerate
            </Button>
          )}
        </div>

        {/* Debug Info */}
        {requestPayload && (
          <Card className="mb-6">
            <CardHeader>
              <CardTitle>🔍 LLM Prompt Info</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Prompt Length:</strong> {requestPayload.prompt.length}{" "}
                  characters
                </p>
                <p>
                  <strong>Available Tools:</strong>{" "}
                  {requestPayload.availableTools.length}
                </p>
                <p>
                  <strong>Model:</strong> {requestPayload.modelConfig.model}
                </p>
                <details className="mt-4">
                  <summary className="cursor-pointer font-medium">
                    View Full Payload JSON
                  </summary>
                  <pre className="mt-2 p-4 bg-muted rounded-lg overflow-x-auto text-xs">
                    {JSON.stringify(requestPayload, null, 2)}
                  </pre>
                </details>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Content */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Plot Results
              {loading && (
                <Loader2 className="h-5 w-5 animate-spin text-primary" />
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="space-y-6">
                <div className="flex flex-col items-center justify-center py-12">
                  <Loader2 className="h-12 w-12 animate-spin text-primary mb-4" />
                  <h3 className="text-lg font-semibold text-foreground mb-2">
                    Generating Baseball Analysis...
                  </h3>
                  <p className="text-sm text-muted-foreground text-center max-w-md">
                    Fetching MLB data and creating visualizations using advanced
                    analytics.
                  </p>
                </div>

                <div className="space-y-4">
                  <Skeleton className="h-8 w-3/4" />
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-5/6" />
                  <div className="grid grid-cols-3 gap-4 mt-6">
                    <Skeleton className="h-24" />
                    <Skeleton className="h-24" />
                    <Skeleton className="h-24" />
                  </div>
                  <Skeleton className="h-48 w-full mt-6" />
                </div>
              </div>
            ) : error ? (
              <div className="text-center py-12">
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-destructive/20 mb-4">
                  <span className="text-destructive text-2xl">⚠️</span>
                </div>
                <h3 className="text-lg font-semibold text-destructive mb-2">
                  Analysis Generation Failed
                </h3>
                <p className="text-sm text-muted-foreground max-w-md mx-auto mb-6">
                  {error}
                </p>
                <div className="flex gap-3 justify-center">
                  <Button variant="outline" onClick={handleBackToHome}>
                    Go Back to Home
                  </Button>
                  <Button onClick={handleRetry}>Try Again</Button>
                </div>
              </div>
            ) : plotResults.length > 0 ? (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {plotResults.map((plot, index) => (
                    <div
                      key={index}
                      className="border rounded-lg overflow-hidden"
                    >
                      <div className="bg-muted px-4 py-2 border-b">
                        <h3 className="text-sm font-medium text-foreground">
                          {plot.plot_title}
                        </h3>
                      </div>
                      <div className="p-4">
                        <img
                          src={`data:image/png;base64,${plot.plot_base64}`}
                          alt={plot.plot_title}
                          className="w-full h-auto rounded"
                        />
                      </div>
                    </div>
                  ))}
                </div>

                <div className="flex justify-between items-center pt-4">
                  <p className="text-sm text-muted-foreground">
                    {plotResults.length} plots generated using MLB data and
                    advanced analytics!
                  </p>
                  <div className="flex gap-2">
                    <Button variant="outline" onClick={handleBackToHome}>
                      Create New Analysis
                    </Button>
                    <Button onClick={handleRetry}>Regenerate Plots</Button>
                  </div>
                </div>
              </div>
            ) : null}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
