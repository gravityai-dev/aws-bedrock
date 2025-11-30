import { createPlugin, type GravityPluginAPI } from "@gravityai-dev/plugin-base";
import packageJson from "../package.json";

const plugin = createPlugin({
  name: packageJson.name,
  version: packageJson.version,
  description: packageJson.description,

  async setup(api: GravityPluginAPI) {
    // Initialize platform dependencies
    const { initializePlatformFromAPI } = await import("@gravityai-dev/plugin-base");
    initializePlatformFromAPI(api);

    // Import and register nodes - API automatically extracts type from definition
    const { BedrockClaudeNode } = await import("./BedrockClaude/node");
    api.registerNode(BedrockClaudeNode);

    const { BedrockEmbeddingNode } = await import("./BedrockEmbedding/node");
    api.registerNode(BedrockEmbeddingNode);

    const { BedrockEmbeddingServiceNode } = await import("./BedrockEmbeddingService/node");
    api.registerNode(BedrockEmbeddingServiceNode);

    // Import and register AWS credential
    const { AWSCredential } = await import("./credentials");
    api.registerCredential(AWSCredential);

    // Services are now part of individual nodes, not registered globally
    // They are called directly by node executors
  },
});

export default plugin;

// Export shared utilities
export { generateEmbedding } from "./shared/embedding";
export type { EmbeddingConfig, AWSCredentials } from "./shared/embedding";
