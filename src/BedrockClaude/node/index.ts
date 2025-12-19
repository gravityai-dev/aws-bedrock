/**
 * Bedrock Claude Node Definition
 * Provides AI text generation capabilities using AWS Bedrock Claude models
 */

import { getPlatformDependencies, type EnhancedNodeDefinition } from "@gravityai-dev/plugin-base";
import BedrockClaudeExecutor from "./executor";

export function createNodeDefinition(): EnhancedNodeDefinition {
  const { NodeInputType } = getPlatformDependencies();

  return {
    packageVersion: "1.1.0",
    type: "BedrockClaude",
    isService: false,
    name: "Bedrock Claude",
    description: "AWS Bedrock Claude models with optional tool support",
    category: "AI",
    color: "#10a37f",
    logoUrl: "https://res.cloudinary.com/sonik/image/upload/v1749137717/gravity/icons/vdyfnijyuyk8ajqtp3nu.webp",

    inputs: [
      {
        name: "signal",
        type: NodeInputType.OBJECT,
        description: "Input to prompt",
      },
    ],

    outputs: [
      {
        name: "output",
        type: NodeInputType.OBJECT,
        description: "Response object",
      },
      {
        name: "usage",
        type: NodeInputType.OBJECT,
        description: "Token burn",
      },
      {
        name: "toolUse",
        type: NodeInputType.OBJECT,
        description: "Selected Tool",
      },
    ],

    configSchema: {
      type: "object",
      properties: {
        model: {
          type: "string",
          title: "Model",
          description: "Select the Claude model to use",
          enum: ["us.anthropic.claude-sonnet-4-5-20250929-v1:0", "us.anthropic.claude-haiku-4-5-20251001-v1:0"],
          enumNames: ["Claude Sonnet 4.5", "Claude 4.5 Haiku"],
          default: "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        },
        maxTokens: {
          type: "number",
          title: "Max Tokens",
          description: "Maximum number of tokens to generate",
          default: 256,
          minimum: 1,
          maximum: 4096,
        },
        temperature: {
          type: "number",
          title: "Temperature",
          description: "Controls randomness (0-1)",
          default: 0.7,
          minimum: 0,
          maximum: 1,
          step: 0.1,
        },
        systemPrompt: {
          type: "string",
          title: "System Prompt",
          description:
            "System message prompt. Supports template syntax like {{input.fieldName}} to reference input data.",
          default: "",
          "ui:field": "template",
        },
        prompt: {
          type: "string",
          title: "Prompt",
          description:
            "User message/prompt. Supports template syntax like {{input.fieldName}} to reference input data.",
          default: "",
          "ui:field": "template",
        },
        includeImageUrl: {
          type: "boolean",
          title: "Include Image URL",
          description: "Enable image analysis by providing an image URL",
          default: false,
          "ui:widget": "toggle",
        },
        imageUrl: {
          type: "string",
          title: "Image URL",
          description: "URL of the image to analyze. Supports template syntax like {{input.imageUrl}}",
          default: "",
          "ui:field": "template",
          "ui:dependencies": {
            includeImageUrl: true,
          },
        },
        enableTools: {
          type: "boolean",
          title: "Enable Tools",
          description: "Enable tool usage for structured outputs",
          default: false,
          "ui:widget": "toggle",
        },
        toolChoice: {
          type: "string",
          title: "Tool Choice",
          description: "How Claude should use tools",
          enum: ["required", "auto"],
          enumNames: ["Required - Must use tools", "Auto - Optional tool use"],
          default: "required",
          "ui:dependencies": {
            enableTools: true,
          },
        },
        toolSchema: {
          type: "object",
          title: "Tool Schema",
          description: "JSON schema for the tool function",
          default: "{}",
          "ui:field": "template",
          "ui:dependencies": {
            enableTools: true,
          },
        },
      },
      required: ["model"],
    },

    credentials: [
      {
        name: "awsCredential",
        required: true,
        displayName: "AWS Credentials",
        description: "AWS credentials for Bedrock API access (accessKeyId, secretAccessKey, region)",
      },
    ],
  };
}

const definition = createNodeDefinition();

export const BedrockClaudeNode = {
  definition,
  executor: BedrockClaudeExecutor,
};

export { definition };
