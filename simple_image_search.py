import sys
sys.path.insert(0, "clip")

from PIL import Image
import mlx.core as mx
import clip


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return (a @ b.T) / (mx.linalg.norm(a) * mx.linalg.norm(b))


def main():
    # Load model
    print("Loading model...")
    model, tokenizer, img_processor = clip.load("clip/mlx_model")

    # Embed images
    image_paths = ["clip/assets/cat.jpeg", "clip/assets/dog.jpeg"]
    print(f"Embedding {len(image_paths)} images...")

    images = [Image.open(p) for p in image_paths]
    pixel_values = img_processor(images)
    image_output = model(pixel_values=pixel_values)
    image_embeds = image_output.image_embeds

    # Search loop
    print("\nReady! Enter a search query (or 'quit' to exit):\n")
    while True:
        query = input("Search: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        # Embed query
        text_tokens = tokenizer([query])
        text_output = model(input_ids=text_tokens)
        text_embed = text_output.text_embeds[0]

        # Find most similar image
        similarities = []
        for i, img_embed in enumerate(image_embeds):
            sim = cosine_similarity(text_embed, img_embed).item()
            similarities.append((sim, image_paths[i]))

        # Sort by similarity (highest first)
        similarities.sort(reverse=True)

        print("\nResults:")
        for sim, path in similarities:
            print(f"  {sim:.4f}  {path}")
        print()


if __name__ == "__main__":
    main()
