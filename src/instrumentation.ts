// Pré-carrega o modelo de embeddings quando o servidor sobe, para a primeira busca não esperar.
export async function register() {
  if (process.env.NEXT_RUNTIME !== "nodejs") return;
  const { warmUpEmbedder } = await import("@/lib/semantic-search");
  warmUpEmbedder().catch((error) => {
    console.error("Falha ao pré-carregar o modelo de embeddings:", error);
  });
}
