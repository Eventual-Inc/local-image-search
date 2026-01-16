"""FastAPI server for image search."""

from pathlib import Path

import daft
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from core import load_model, embed_text, cosine_similarity, DB_PATH

app = FastAPI(title="Local Image Search")

# Global state - loaded on startup
model = None
tokenizer = None
embeddings_df = None


class SearchRequest(BaseModel):
    query: str
    limit: int = 10


class SearchResult(BaseModel):
    path: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total_images: int


@app.on_event("startup")
async def startup():
    """Load model and embeddings on startup."""
    global model, tokenizer, embeddings_df

    print("Loading CLIP model...")
    model, tokenizer, _ = load_model()

    print("Loading embeddings...")
    if Path(DB_PATH).exists():
        embeddings_df = daft.read_lance(DB_PATH).collect()
        print(f"Loaded {len(embeddings_df)} embeddings")
    else:
        print("No embeddings found. Run embed.py first.")
        embeddings_df = None


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "embeddings_loaded": embeddings_df is not None}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search for images matching the query."""
    if embeddings_df is None:
        return SearchResponse(results=[], total_images=0)

    # Embed the query text
    query_embedding = embed_text(model, tokenizer, request.query)

    # Get all embeddings and paths
    data = embeddings_df.to_pydict()
    paths = data["path"]
    vectors = data["vector"]

    # Compute similarities
    scores = []
    for i, vec in enumerate(vectors):
        vec_array = np.array(vec, dtype=np.float32)
        # Skip zero vectors (failed images)
        if np.allclose(vec_array, 0):
            scores.append(-1.0)
        else:
            scores.append(cosine_similarity(query_embedding, vec_array))

    # Sort by score descending
    ranked = sorted(zip(paths, scores), key=lambda x: x[1], reverse=True)

    # Return top results
    results = [
        SearchResult(path=path, score=score)
        for path, score in ranked[:request.limit]
        if score > 0  # exclude failed images
    ]

    return SearchResponse(results=results, total_images=len(paths))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
