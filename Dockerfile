# Sonar Municipal — imagem do app Next.js (site + API) com o modelo de embeddings embutido.

FROM node:24-bookworm-slim AS build
WORKDIR /app
# O app roda em CPU: não baixa os binários CUDA do onnxruntime-node.
ENV ONNXRUNTIME_NODE_INSTALL=skip \
    NEXT_TELEMETRY_DISABLED=1
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run build

FROM node:24-bookworm-slim AS runner
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    ONNXRUNTIME_NODE_INSTALL=skip \
    EMBEDDING_MODEL_ID=Xenova/multilingual-e5-base \
    EMBEDDING_CACHE_DIR=/app/models
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --no-audit --no-fund && npm cache clean --force

# Baixa o modelo durante o build: em execução o servidor não depende do Hugging Face.
RUN node -e "import('@huggingface/transformers').then(async ({ env, pipeline }) => { \
      env.cacheDir = process.env.EMBEDDING_CACHE_DIR; \
      await pipeline('feature-extraction', process.env.EMBEDDING_MODEL_ID, { dtype: 'fp32' }); \
    })"

COPY --from=build /app/next.config.ts ./
COPY --from=build /app/public ./public
COPY --from=build /app/indicators ./indicators
COPY --from=build --chown=node:node /app/.next ./.next

USER node
EXPOSE 3000
CMD ["node_modules/.bin/next", "start", "-H", "0.0.0.0", "-p", "3000"]
