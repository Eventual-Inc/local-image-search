"""Image search using Daft for batch embedding."""

import daft
from daft import col

from core import EmbedImages, load_model, embed_text, cosine_similarity


def main():
    # Create DataFrame with image paths
    image_paths = ["clip/assets/cat.jpeg", "clip/assets/dog.jpeg"]

    print("Creating DataFrame...")
    df = daft.from_pydict({"path": image_paths})
    df.show()

    # Generate embeddings using UDF
    print("\nGenerating embeddings with Daft UDF...")
    embed_images = EmbedImages()
    df = df.with_column("embedding", embed_images(col("path")))
    df.show()

    # Collect results for searching
    results = df.collect()
    paths = results.to_pydict()["path"]
    embeddings = results.to_pydict()["embedding"]

    # Load model for text embedding
    print("\nLoading model for text queries...")
    model, tokenizer, _ = load_model()

    # Search loop
    print("\nReady! Enter a search query (or 'quit' to exit):\n")
    while True:
        query = input("Search: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        # Embed query
        query_embed = embed_text(model, tokenizer, query)

        # Compute similarities
        similarities = []
        for path, emb in zip(paths, embeddings):
            sim = cosine_similarity(query_embed, emb)
            similarities.append((sim, path))

        # Sort by similarity (highest first)
        similarities.sort(reverse=True)

        print("\nResults:")
        for sim, path in similarities:
            print(f"  {sim:.4f}  {path}")
        print()


if __name__ == "__main__":
    main()
