import { getPlatformDependencies, type EnhancedNodeDefinition } from "@gravityai-dev/plugin-base";
import BedrockEmbeddingServiceExecutor from "./executor";

/**
 * BedrockEmbeddingService - Dedicated Service Node
 *
 * This is a PURE SERVICE NODE that provides embedding services to other nodes.
 * It is NOT part of workflow execution - it responds to serviceConnector calls.
 */
export function createNodeDefinition(): EnhancedNodeDefinition {
  const { NodeInputType } = getPlatformDependencies();

  return {
    packageVersion: "1.0.40",
    type: "BedrockEmbeddingService",
    name: "Embedding Service",
    description: "AWS Bedrock embedding service provider - responds to SERVICE_CALL signals",
    category: "AI",
    color: "#10a37f",
    logoUrl: "https://res.cloudinary.com/sonik/image/upload/v1749137717/gravity/icons/vdyfnijyuyk8ajqtp3nu.webp",

    // Node template for styling
    template: "service", // Options: "standard", "service", "mini"

    // NO REGULAR INPUTS/OUTPUTS - services use service connectors
    inputs: [],
    outputs: [],

    // SERVICE CONNECTORS - defines what services this node provides
    serviceConnectors: [
      {
        name: "embeddingService",
        description: "Provides embedding generation services",
        serviceType: "embedding",
        methods: ["createEmbedding", "createBatchEmbeddings"],
        isService: true, // This connector PROVIDES embedding services to others
      },
    ],

    // Configuration for service
    configSchema: {
      type: "object",
      properties: {
        model: {
          type: "string",
          title: "Model",
          description: "Select the Bedrock embedding model to use",
          enum: ["amazon.titan-embed-text-v2:0", "amazon.titan-embed-text-v1"],
          enumNames: ["Titan Text Embeddings v2", "Titan Text Embeddings v1"],
          default: "amazon.titan-embed-text-v2:0",
        },
        normalize: {
          type: "boolean",
          title: "Normalize Embeddings",
          description: "Whether to normalize the embedding vectors (recommended for similarity search)",
          default: true,
          "ui:widget": "toggle",
        },
        dimensions: {
          type: "number",
          title: "Output Dimensions",
          description: "Number of dimensions for the output embedding",
          enum: [256, 512, 1024],
          enumNames: ["256 dimensions", "512 dimensions", "1024 dimensions"],
          default: 1024,
        },
      },
      required: ["model"],
    },

    credentials: [
      {
        name: "awsCredential",
        required: true,
        displayName: "AWS Credentials",
        description: "AWS credentials for accessing Bedrock API",
      },
    ],
  };
}

const definition = createNodeDefinition();

export const BedrockEmbeddingServiceNode = {
  definition,
  executor: BedrockEmbeddingServiceExecutor,
};

export { definition };
