"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useMCP, type Prompt } from "@/lib/mcp-context";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Loader2, ArrowDown } from "lucide-react";

export default function Home() {
  const router = useRouter();
  const { client: mcpClient, prompts, loading, error } = useMCP();
  const [selectedPrompt, setSelectedPrompt] = useState<string>("");
  const [argumentValues, setArgumentValues] = useState<Record<string, string>>(
    {}
  );
  const [completePromptText, setCompletePromptText] = useState<string>("");
  const [selectedModel, setSelectedModel] = useState<string>("gpt-4o-mini");

  const filteredPrompts = prompts.filter((prompt: Prompt) =>
    prompt.name.startsWith("web_ui_")
  );

  const getDisplayName = (promptName: string) => {
    return promptName.startsWith("web_ui_")
      ? promptName.substring(7) // Remove "web_ui_" prefix
      : promptName;
  };

  const handlePromptChange = (value: string) => {
    setSelectedPrompt(value);
    setArgumentValues({});
    setCompletePromptText("");
  };

  const handleArgumentChange = (argName: string, value: string) => {
    const newArgValues = { ...argumentValues, [argName]: value };
    setArgumentValues(newArgValues);

    // Update complete prompt text when arguments change
    if (selectedPromptData && selectedPrompt !== "custom") {
      updateCompletePromptText(selectedPromptData, newArgValues);
    }
  };

  const updateCompletePromptText = async (
    promptData: Prompt,
    argValues: Record<string, string>
  ) => {
    if (!mcpClient) return;

    try {
      // Check if all required arguments are filled
      const allRequiredFilled = (promptData.arguments || []).every(
        (arg: { name: string; required?: boolean }) =>
          !arg.required ||
          (argValues[arg.name] && argValues[arg.name].trim() !== "")
      );

      if (!allRequiredFilled) {
        setCompletePromptText("");
        return;
      }

      // Use MCP client to get the completed prompt
      const result = await mcpClient.getPrompt({
        name: promptData.name,
        arguments: argValues,
      });

      // Extract the prompt text from the result
      if (result && result.messages && result.messages.length > 0) {
        const promptText = result.messages
          .map((msg) => msg.content?.text || JSON.stringify(msg.content))
          .join("\n");
        setCompletePromptText(promptText);
      } else {
        // Fallback if the response format is different
        setCompletePromptText(JSON.stringify(result, null, 2));
      }
    } catch (error) {
      console.error("Error completing prompt:", error);
      setCompletePromptText(
        `Error completing prompt: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  };

  const selectedPromptData =
    selectedPrompt === "custom"
      ? null
      : filteredPrompts.find(
          (p: Prompt) => p.name === "web_ui_" + selectedPrompt
        );

  // Check if all required arguments are filled
  const allRequiredArgsFilled = selectedPromptData
    ? (selectedPromptData.arguments || []).every(
        (arg: { name: string; required?: boolean }) =>
          !arg.required ||
          (argumentValues[arg.name] && argumentValues[arg.name].trim() !== "")
      )
    : false;

  const showPromptTextCard =
    selectedPrompt === "custom" ||
    (selectedPromptData &&
      allRequiredArgsFilled &&
      completePromptText.trim() !== "");

  const scrollToBottom = () => {
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  };

  const handleSubmit = () => {
    console.log("Submitting prompt:", completePromptText);

    if (!completePromptText.trim()) {
      console.error("No prompt text to submit");
      return;
    }

    if (!mcpClient) {
      console.error("MCP client not available");
      return;
    }

    // Store data in sessionStorage for the results page
    const submissionData = {
      completePromptText,
      model: selectedModel,
    };

    sessionStorage.setItem(
      "baseball-submission-data",
      JSON.stringify(submissionData)
    );

    // Navigate to results page
    router.push("/results");
  };

  return (
    <div className="dark min-h-screen bg-background text-foreground py-8">
      <div className="max-w-4xl mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-foreground mb-2">
            AI Baseball Analyst ⚾️
          </h1>
          <p className="text-muted-foreground">
            Generate statistical reports about baseball using an LLM enabled
            with Model Context Protocol
          </p>
        </div>

        {/* Content */}
        <Card>
          <CardHeader>
            <CardTitle>Connection Status</CardTitle>
            <div className="flex items-center gap-2">
              <Badge
                variant="outline"
                className="flex items-center gap-2 text-sm"
              >
                <div
                  className={`w-2 h-2 rounded-full ${
                    loading
                      ? "bg-muted-foreground"
                      : error
                      ? "bg-destructive"
                      : "bg-green-500"
                  }`}
                ></div>
                {loading ? "Establishing" : error ? "Failed" : "Connected"}
              </Badge>
              {!loading && !error && (
                <Badge
                  variant="secondary"
                  className="flex items-center gap-2 text-sm bg-white text-black border"
                >
                  ({filteredPrompts.length}) prompts available
                </Badge>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex flex-col items-center justify-center py-8">
                <Loader2 className="h-8 w-8 animate-spin text-primary mb-3" />
                <p className="text-muted-foreground">
                  Connecting to MCP server...
                </p>
              </div>
            ) : error ? (
              <div className="text-center py-8">
                <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-destructive/20 mb-4">
                  <span className="text-destructive text-xl">⚠️</span>
                </div>
                <h3 className="text-lg font-semibold text-destructive mb-2">
                  Connection Error
                </h3>
                <p className="text-sm text-muted-foreground max-w-md mx-auto">
                  {error}
                </p>
              </div>
            ) : (
              <div className="space-y-6">
                <div className="space-y-4">
                  <label className="text-lg font-semibold text-foreground">
                    Select a Prompt
                  </label>
                  <Select
                    value={selectedPrompt}
                    onValueChange={handlePromptChange}
                  >
                    <SelectTrigger className="h-auto min-h-14 py-4">
                      <SelectValue placeholder="Choose a prompt..." />
                    </SelectTrigger>
                    <SelectContent className="max-h-80">
                      {filteredPrompts.map((prompt, index) => (
                        <SelectItem
                          key={index}
                          value={getDisplayName(prompt.name)}
                        >
                          <div className="flex flex-col items-start justify-center py-2 min-h-12">
                            <span className="font-medium">
                              {getDisplayName(prompt.name)}
                            </span>
                            {prompt.description && (
                              <span className="text-xs text-muted-foreground mt-1 max-w-xs truncate">
                                {prompt.description}
                              </span>
                            )}
                          </div>
                        </SelectItem>
                      ))}
                      <SelectItem value="custom">
                        <div className="flex flex-col items-start justify-center py-2 min-h-12 border-t pt-4 mt-2">
                          <span className="font-medium">Custom Prompt</span>
                          <span className="text-xs text-muted-foreground mt-1">
                            Advanced: Create your own custom prompt
                          </span>
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {filteredPrompts.length === 0 && (
                  <div className="text-center py-4">
                    <p className="text-sm text-muted-foreground">
                      No prompts available from the MCP server.
                    </p>
                  </div>
                )}

                {selectedPromptData && (
                  <Card>
                    <CardHeader className="pb-3">
                      <div className="flex items-center gap-2">
                        <CardTitle className="text-primary text-lg">
                          {selectedPromptData.name}
                        </CardTitle>
                        <Badge variant="secondary">Selected</Badge>
                      </div>
                      {selectedPromptData.description && (
                        <CardDescription>
                          {selectedPromptData.description}
                        </CardDescription>
                      )}
                    </CardHeader>
                    {selectedPromptData.arguments &&
                      selectedPromptData.arguments.length > 0 && (
                        <CardContent className="pt-0">
                          <div className="space-y-4">
                            <h4 className="text-sm font-medium text-foreground">
                              Arguments:
                            </h4>
                            <div className="space-y-4">
                              {selectedPromptData.arguments.map(
                                (
                                  arg: {
                                    name: string;
                                    description?: string;
                                    required?: boolean;
                                  },
                                  index: number
                                ) => (
                                  <div key={index} className="space-y-2">
                                    <div className="flex items-center gap-2">
                                      <Badge
                                        variant={
                                          arg.required
                                            ? "destructive"
                                            : "outline"
                                        }
                                        className="text-xs"
                                      >
                                        {arg.name}
                                      </Badge>
                                      {arg.required && (
                                        <span className="text-xs text-destructive">
                                          required
                                        </span>
                                      )}
                                    </div>
                                    {arg.description && (
                                      <p className="text-xs text-muted-foreground">
                                        {arg.description}
                                      </p>
                                    )}
                                    <Input
                                      placeholder={`Enter ${arg.name}...`}
                                      value={argumentValues[arg.name] || ""}
                                      onChange={(e) =>
                                        handleArgumentChange(
                                          arg.name,
                                          e.target.value
                                        )
                                      }
                                    />
                                  </div>
                                )
                              )}
                            </div>
                          </div>
                        </CardContent>
                      )}
                  </Card>
                )}

                {showPromptTextCard && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-primary">
                        Complete Prompt
                      </CardTitle>
                      <CardDescription>
                        Review and edit your complete prompt before submitting
                      </CardDescription>
                      <div className="flex gap-2">
                        <Button
                          onClick={scrollToBottom}
                          variant="outline"
                          className="flex items-center gap-2"
                        >
                          <ArrowDown className="h-4 w-4" />
                          Scroll to Submit
                        </Button>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <Textarea
                        placeholder="Enter your custom prompt here..."
                        value={completePromptText}
                        onChange={(e) => setCompletePromptText(e.target.value)}
                        className="min-h-32 resize-none"
                      />
                      <div className="flex gap-2 items-end">
                        <div className="flex-1">
                          <label className="text-sm font-medium text-foreground mb-2 block">
                            Select Model
                          </label>
                          <Select
                            value={selectedModel}
                            onValueChange={setSelectedModel}
                          >
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="gpt-4.1-mini">
                                GPT-4.1 Mini
                              </SelectItem>
                              <SelectItem value="gpt-4.1-nano">
                                GPT-4.1 Nano
                              </SelectItem>
                              <SelectItem value="gpt-4o-mini">
                                GPT-4o Mini
                              </SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                        <Button
                          onClick={handleSubmit}
                          className="bg-destructive hover:bg-destructive/90 text-destructive-foreground px-8"
                          disabled={!completePromptText.trim()}
                        >
                          Submit
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
