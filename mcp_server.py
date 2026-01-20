#!/usr/bin/env python3
"""MCP server for local image search."""

from pathlib import Path

import daft
import numpy as np
from mcp.server.fastmcp import FastMCP

from core import load_model, embed_text, cosine_similarity, DB_PATH

# Create MCP server
mcp = FastMCP("local-image-search")

# Global state - loaded on startup
model = None
tokenizer = None
embeddings_df = None


@mcp.tool()
def search_images(query: str, limit: int = 5) -> list[dict]:
    """Search for images matching a text query.

    Args:
        query: Natural language description of the image to find
        limit: Maximum number of results to return (default: 5)

    Returns:
        List of matching images with paths and similarity scores
    """
    global model, tokenizer, embeddings_df

    if embeddings_df is None:
        return [{"error": "No embeddings loaded. Run embed.py first."}]

    # Embed the query text
    query_embedding = embed_text(model, tokenizer, query)

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
        {"path": path, "score": round(score, 3)}
        for path, score in ranked[:limit]
        if score > 0  # exclude failed images
    ]

    return results


def main():
    """Main entry point."""
    global model, tokenizer, embeddings_df

    print("Loading CLIP model...", flush=True)
    model, tokenizer, _ = load_model()

    print("Loading embeddings...", flush=True)
    if Path(DB_PATH).exists():
        embeddings_df = daft.read_lance(DB_PATH).collect()
        print(f"Loaded {len(embeddings_df)} embeddings", flush=True)
    else:
        print("No embeddings found. Run embed.py first.", flush=True)

    # Run the MCP server
    mcp.run()


if __name__ == "__main__":
    main()
