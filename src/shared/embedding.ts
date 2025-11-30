/**
 * Shared embedding utility
 * Can be used by both nodes and internal services without requiring service connectors
 */

import { BedrockRuntimeClient, InvokeModelCommand } from "@aws-sdk/client-bedrock-runtime";
import { getPlatformDependencies } from "@gravityai-dev/plugin-base";

export interface EmbeddingConfig {
  model?: string;
  dimensions?: number;
  normalize?: boolean;
}

export interface AWSCredentials {
  accessKeyId: string;
  secretAccessKey: string;
  region?: string;
}

/**
 * Generate embedding using AWS Bedrock
 * Can be called directly without node execution context
 */
export async function generateEmbedding(
  text: string,
  credentials: AWSCredentials,
  config: EmbeddingConfig = {}
): Promise<number[]> {
  const { createLogger } = getPlatformDependencies();
  const logger = createLogger("BedrockEmbedding");

  const client = new BedrockRuntimeClient({
    region: credentials.region || 'us-east-1',
    credentials: {
      accessKeyId: credentials.accessKeyId,
      secretAccessKey: credentials.secretAccessKey,
    },
  });

  const model = config.model || 'amazon.titan-embed-text-v2:0';
  const dimensions = config.dimensions || 1024;
  const normalize = config.normalize !== false;

  const requestBody = {
    inputText: text,
    dimensions,
    normalize,
  };

  logger.debug('Generating embedding', {
    model,
    dimensions,
    textLength: text.length,
  });

  const command = new InvokeModelCommand({
    modelId: model,
    body: JSON.stringify(requestBody),
  });

  const response = await client.send(command);
  const result = JSON.parse(new TextDecoder().decode(response.body));

  logger.debug('Embedding generated', {
    embeddingLength: result.embedding?.length,
  });

  return result.embedding;
}
