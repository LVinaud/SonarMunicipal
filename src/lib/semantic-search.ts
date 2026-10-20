import { env, pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";
import { QdrantClient } from "@qdrant/js-client-rest";

// Mesmo modelo da indexação (multilingual-e5-base), em ONNX fp32, executado localmente na CPU.
const EMBEDDING_MODEL_ID = process.env.EMBEDDING_MODEL_ID ?? "Xenova/multilingual-e5-base";
if (process.env.EMBEDDING_CACHE_DIR) {
  env.cacheDir = process.env.EMBEDDING_CACHE_DIR;
}
const QDRANT_COLLECTION = process.env.QDRANT_COLLECTION ?? "projetos-de-lei";
// Busca exata (força bruta) por padrão: neste corpus o índice HNSW perde resultados relevantes
// (recall de ~70% no top-10 em testes), e a varredura completa custa ~50 ms por consulta.
const EXACT_SEARCH = process.env.QDRANT_EXACT_SEARCH !== "false";
const MAX_TOP_K = (() => {
  const raw = Number.parseInt(process.env.SEARCH_MAX_RESULTS ?? "", 10);
  if (Number.isFinite(raw) && raw > 0) return raw;
  return 1000;
})();

type SearchParams = {
  query: string;
  limit?: number;
};

type Payload = Record<string, unknown> | null | undefined;

export type SearchHit = {
  index: number;
  score: number;
  municipio?: string;
  uf?: string;
  acao?: string;
  data_apresentacao?: string;
  ementa?: string;
  link_publico?: string | null;
  sapl_url?: string | null;
  tipo_label?: string | null;
};

let embedderPromise: Promise<FeatureExtractionPipeline> | null = null;
let qdrantClient: QdrantClient | null = null;

const ensureNumberArray = (value: unknown): number[] => {
  if (Array.isArray(value)) {
    if (value.length > 0 && Array.isArray(value[0])) {
      return ensureNumberArray(value[0]);
    }
    return (value as unknown[]).map((item) => Number(item));
  }
  if (typeof value === "number") return [value];
  return [];
};

const normalizeVector = (vector: number[]): number[] => {
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  if (!Number.isFinite(norm) || norm === 0) return vector;
  return vector.map((value) => value / norm);
};

const getString = (payload: Payload, key: string): string | undefined => {
  if (!payload || typeof payload !== "object") return undefined;
  const value = (payload as Record<string, unknown>)[key];
  return typeof value === "string" ? value : undefined;
};

const getNumberId = (id: unknown): number => {
  if (typeof id === "number" && Number.isFinite(id)) return id;
  if (typeof id === "string") {
    const parsed = Number.parseInt(id, 10);
    if (Number.isFinite(parsed)) return parsed;
  }
  return -1;
};

const toSearchHit = (item: { id: unknown; score?: number; payload?: Payload }): SearchHit => {
  const payload = item.payload as Payload;
  return {
    index: getNumberId(item.id),
    score: item.score ?? 0,
    municipio: getString(payload, "municipio"),
    uf: getString(payload, "uf"),
    acao: getString(payload, "acao"),
    data_apresentacao: getString(payload, "data_apresentacao"),
    ementa: getString(payload, "ementa"),
    link_publico: getString(payload, "link_publico"),
    sapl_url: getString(payload, "sapl_url"),
    tipo_label: getString(payload, "tipo_label"),
  };
};

const getEmbedder = (): Promise<FeatureExtractionPipeline> => {
  if (!embedderPromise) {
    embedderPromise = pipeline("feature-extraction", EMBEDDING_MODEL_ID, { dtype: "fp32" }).catch((error) => {
      embedderPromise = null;
      throw error;
    });
  }
  return embedderPromise;
};

export const warmUpEmbedder = async (): Promise<void> => {
  await embedQuery("aquecimento");
};

const getQdrantClient = (): QdrantClient => {
  if (qdrantClient) return qdrantClient;
  const url = process.env.QDRANT_URL;
  if (!url) {
    throw new Error("Env QDRANT_URL não definido.");
  }
  qdrantClient = new QdrantClient({ url, apiKey: process.env.QDRANT_API_KEY || undefined });
  return qdrantClient;
};

export const embedQuery = async (query: string): Promise<number[]> => {
  const text = query.trim();
  if (!text) {
    throw new Error("Query vazia.");
  }

  // O pipeline trunca à direita no limite do modelo (512 tokens), como a Inference API fazia.
  const embedder = await getEmbedder();
  const output = await embedder(`query: ${text}`, { pooling: "mean", normalize: true });

  const vector = normalizeVector(ensureNumberArray(output.tolist()));
  if (vector.length === 0) {
    throw new Error("Falha ao gerar embedding para a query.");
  }
  return vector;
};

export const searchProjects = async ({ query, limit }: SearchParams): Promise<SearchHit[]> => {
  const vector = await embedQuery(query);
  const client = getQdrantClient();
  const safeLimit = Math.max(1, Math.min(limit ?? 50, MAX_TOP_K));

  const results = await client.search(QDRANT_COLLECTION, {
    vector,
    limit: safeLimit,
    with_payload: true,
    with_vector: false,
    params: { exact: EXACT_SEARCH },
  });

  return results.map((item) => toSearchHit(item));
};

export const fetchProjectsByIds = async (ids: Array<number | string>): Promise<SearchHit[]> => {
  if (!ids.length) return [];
  const client = getQdrantClient();
  const unique = Array.from(new Set(ids)).slice(0, MAX_TOP_K);

  const items = await client.retrieve(QDRANT_COLLECTION, {
    ids: unique,
    with_payload: true,
    with_vector: false,
  });

  return items.map((item) =>
    toSearchHit({
      id: item.id,
      payload: item.payload,
      score: (item as { score?: number }).score ?? 0,
    }),
  );
};
