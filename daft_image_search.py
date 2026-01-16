import sys
sys.path.insert(0, "clip")

import daft
from daft import col, DataType, Series
from PIL import Image
import numpy as np
import clip


@daft.cls
class EmbedImages:
    """UDF to generate CLIP embeddings for images."""

    def __init__(self):
        self.model, _, self.img_processor = clip.load("clip/mlx_model")

    @daft.method.batch(return_dtype=DataType.embedding(DataType.float32(), 512))
    def __call__(self, paths: Series):
        """Takes a Series of image paths, returns a list of embeddings (one 512-dim float32 array per image)."""
        images = [Image.open(p) for p in paths.to_pylist()]
        pixel_values = self.img_processor(images)
        output = self.model(pixel_values=pixel_values)
        # Convert MLX array to list of numpy arrays
        embeddings = [np.array(emb) for emb in output.image_embeds]
        return embeddings


def embed_text(model, tokenizer, text: str) -> np.ndarray:
    """Embed a text query."""
    tokens = tokenizer([text])
    output = model(input_ids=tokens)
    return np.array(output.text_embeds[0])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


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
    model, tokenizer, _ = clip.load("clip/mlx_model")

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
