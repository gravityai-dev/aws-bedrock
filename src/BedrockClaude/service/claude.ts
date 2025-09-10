/**
 * AWS Bedrock Claude service
 * Handles Claude model interactions via AWS Bedrock
 */
import {
  BedrockRuntimeClient,
  ConverseCommand,
  ContentBlock,
  Message,
  Tool,
  ToolConfiguration,
} from "@aws-sdk/client-bedrock-runtime";
import { initializeBedrockClient } from "../../shared/client";
import { getPlatformDependencies } from "@gravityai-dev/plugin-base";

export interface ClaudeMessage {
  role: string;
  content: Array<{ text?: string; image?: { format: string; source: { bytes?: string | Buffer } } }>;
}

export interface ClaudeToolSchema {
  name: string;
  description: string;
  inputSchema: {
    json: {
      type: string;
      properties: Record<string, any>;
      required?: string[];
    };
  };
}

export interface BedrockClaudeConfig {
  model: string;
  temperature: number;
  maxTokens: number;
  systemPrompt?: string;
  prompt: string;
  includeImageUrl?: boolean;
  imageUrl?: string;
  enableTools: boolean;
  toolChoice?: string;
  toolSchema?: string | object;
}

export interface BedrockClaudeServiceResponse {
  text: string;
  usage?: {
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
  };
  toolUse?: {
    toolName: string;
    toolInput: any;
  };
}

/**
 * Call Claude model via AWS Bedrock
 */
export async function callBedrockClaude(
  config: BedrockClaudeConfig,
  credentialContext: any,
  nodeLogger?: any,
  executionContext?: { workflowId: string; executionId: string; nodeId: string }
): Promise<BedrockClaudeServiceResponse> {
  const { createLogger } = getPlatformDependencies();
  const log = nodeLogger || createLogger("BedrockClaude");

  // Initialize client
  const client: BedrockRuntimeClient = await initializeBedrockClient(credentialContext);

  // Prepare messages
  const messageContent: any[] = [];
  
  // Add image if URL is provided
  if (config.includeImageUrl && config.imageUrl) {
    log.info("Including image URL in message", { imageUrl: config.imageUrl });
    
    try {
      const response = await fetch(config.imageUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch image: ${response.status} ${response.statusText}`);
      }
      
      const arrayBuffer = await response.arrayBuffer();
      const bytes = Buffer.from(arrayBuffer);
      
      log.info("Image fetched successfully", {
        imageUrl: config.imageUrl,
        bytesLength: bytes.length,
        status: response.status
      });
      
      // Determine image format from URL or content-type
      const contentType = response.headers.get('content-type') || '';
      let format = 'jpeg'; // default
      if (contentType.includes('png') || config.imageUrl.toLowerCase().includes('.png')) {
        format = 'png';
      } else if (contentType.includes('webp') || config.imageUrl.toLowerCase().includes('.webp')) {
        format = 'webp';
      } else if (contentType.includes('gif') || config.imageUrl.toLowerCase().includes('.gif')) {
        format = 'gif';
      }
      
      log.info("Image format detected", {
        contentType,
        detectedFormat: format
      });
      
      messageContent.push({
        image: {
          format: format,
          source: {
            bytes: bytes
          }
        }
      });
    } catch (error) {
      log.error("Failed to fetch image from URL", { imageUrl: config.imageUrl, error });
      throw new Error(`Failed to fetch image from URL: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  
  // Add text prompt
  messageContent.push({ text: config.prompt });
  
  const messages: ClaudeMessage[] = [{ role: "user", content: messageContent }];

  // Create the request configuration
  const commandInput: any = {
    modelId: config.model,
    messages: messages,
    inferenceConfig: {
      temperature: config.temperature,
      maxTokens: config.maxTokens,
    },
  };

  // Add system prompt if provided
  if (config.systemPrompt) {
    commandInput.system = [{ text: config.systemPrompt }];
  }

  // Add tool configuration if enabled
  if (config.enableTools && config.toolSchema) {
    log.info("Tool configuration enabled", {
      enableTools: config.enableTools,
      toolSchemaRaw: config.toolSchema,
      toolChoice: config.toolChoice,
    });

    try {
      // Parse toolSchema if it's a string
      const tools = typeof config.toolSchema === "string" ? JSON.parse(config.toolSchema) : config.toolSchema;

      // Ensure it's an array
      const toolsArray = Array.isArray(tools) ? tools : [tools];

      log.info("Parsed tools", {
        toolCount: toolsArray.length,
        firstToolName: toolsArray[0]?.name,
      });

      // Only add tools if we have valid tools
      if (toolsArray.length > 0) {
        commandInput.toolConfig = {
          tools: toolsArray,
        };

        // Set tool choice based on config
        if (config.toolChoice === "required") {
          // Always use 'any' for required - Claude must use at least one tool
          commandInput.toolConfig.toolChoice = { any: {} };
        } else if (config.toolChoice === "auto") {
          // Auto - Claude decides whether to use tools
          commandInput.toolConfig.toolChoice = { auto: {} };
        }
      }
    } catch (error) {
      log.warn("Failed to configure tools", {
        error: error instanceof Error ? error.message : String(error),
      });
    }
  } else {
    log.info("Tools not enabled or no schema", {
      enableTools: config.enableTools,
      hasToolSchema: !!config.toolSchema,
    });
  }

  log.info("Calling Bedrock Claude API", {
    model: config.model,
    temperature: config.temperature,
    maxTokens: config.maxTokens,
    enableTools: config.enableTools,
    hasToolSchema: !!config.toolSchema,
    toolSchemaType: typeof config.toolSchema,
    hasToolConfig: !!commandInput.toolConfig,
  });

  try {
    // Send request to Bedrock (non-streaming)
    const response = await client.send(new ConverseCommand(commandInput));

    // Extract the response
    const output = response.output as any;

    if (!output) {
      throw new Error("No output received from Bedrock");
    }

    // Initialize result
    const result: BedrockClaudeServiceResponse = {
      text: "",
    };

    // Handle message response
    if (output.message?.content) {
      for (const contentBlock of output.message.content) {
        if (contentBlock.text) {
          result.text += contentBlock.text;
        } else if (contentBlock.toolUse) {
          // Handle tool use response
          result.toolUse = {
            toolName: contentBlock.toolUse.name,
            toolInput: contentBlock.toolUse.input,
          };
        }
      }
    }

    // Add usage information if available
    if (response.usage) {
      result.usage = {
        inputTokens: response.usage.inputTokens || 0,
        outputTokens: response.usage.outputTokens || 0,
        totalTokens: response.usage.totalTokens || 0,
      };
    }

    log.info("Bedrock Claude API call successful", {
      textLength: result.text.length,
      hasToolUse: !!result.toolUse,
      usage: result.usage,
    });

    return result;
  } catch (error) {
    log.error("Bedrock Claude API call failed", {
      error: error instanceof Error ? error.message : String(error),
    });
    throw error;
  }
}
