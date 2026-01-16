import sys
sys.path.insert(0, "clip")

import time
import daft
from daft import col, DataType, Series
from PIL import Image
import numpy as np
import clip
from pathlib import Path


@daft.cls
class EmbedImages:
    def __init__(self):
        self.model, _, self.img_processor = clip.load("clip/mlx_model")

    @daft.method.batch(return_dtype=DataType.embedding(DataType.float32(), 512))
    def __call__(self, paths: Series):
        """Takes a Series of image paths, returns a list of embeddings."""
        images = [Image.open(p) for p in paths.to_pylist()]
        pixel_values = self.img_processor(images)
        output = self.model(pixel_values=pixel_values)
        return [np.array(emb) for emb in output.image_embeds]


def benchmark(n_images: int):
    """Benchmark embedding n_images."""
    image_dir = Path("data/pokemon")
    image_paths = sorted([str(p) for p in image_dir.glob("*.png")])[:n_images]
    print(f"Testing with {len(image_paths)} image(s)...")

    df = daft.from_pydict({"path": image_paths})
    embed_images = EmbedImages()

    start = time.time()
    df = df.with_column("embedding", embed_images(col("path")))
    results = df.collect()
    elapsed = time.time() - start

    print(f"Time: {elapsed:.2f}s")
    return elapsed


def run_all_benchmarks():
    """Run benchmarks for all data points and save to CSV."""
    data_points = [1, 25, 225, 425, 625, 825, 1025]
    results = []

    for n in data_points:
        elapsed = benchmark(n)
        results.append({"images": n, "time_seconds": elapsed})

    # Save to CSV
    import csv
    with open("benchmark_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["images", "time_seconds"])
        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to benchmark_results.csv")
    print("\nSummary:")
    for r in results:
        print(f"  {r['images']:>4} images: {r['time_seconds']:.2f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        benchmark(n)
    else:
        run_all_benchmarks()
