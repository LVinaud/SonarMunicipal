import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    status: "ok",
    embedding_model: process.env.EMBEDDING_MODEL_ID ?? "Xenova/multilingual-e5-base",
    qdrant_collection: process.env.QDRANT_COLLECTION ?? "projetos-de-lei",
    has_qdrant_url: Boolean(process.env.QDRANT_URL),
  });
}
