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
    """Run one benchmark iteration and append to CSV."""
    import csv

    data_points = [1, 25, 225, 425, 625, 825, 1025]
    csv_path = Path("benchmark_results.csv")

    # Load existing data or initialize
    if csv_path.exists():
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            existing_data = {int(row["images"]): row for row in reader}
        # Determine next run number
        run_cols = [f for f in fieldnames if f.startswith("run_")]
        next_run = len(run_cols) + 1
    else:
        existing_data = {}
        fieldnames = ["images"]
        next_run = 1

    run_col = f"run_{next_run}"
    fieldnames.append(run_col)

    print(f"\n=== Run {next_run} ===\n")

    # Run benchmarks
    for n in data_points:
        elapsed = benchmark(n)
        if n in existing_data:
            existing_data[n][run_col] = elapsed
        else:
            existing_data[n] = {"images": n, run_col: elapsed}

    # Save to CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for n in data_points:
            writer.writerow(existing_data[n])

    print(f"\nResults saved to benchmark_results.csv (run {next_run})")
    print("\nSummary:")
    for n in data_points:
        print(f"  {n:>4} images: {existing_data[n][run_col]:.2f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        benchmark(n)
    else:
        run_all_benchmarks()
