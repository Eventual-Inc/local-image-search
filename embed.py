#!/usr/bin/env python3
"""CLI tool to find and embed images from a directory."""

import argparse
import sys
import time
from pathlib import Path

import daft
from daft import col

from core import EmbedImages, find_images, format_time, IMAGES_PER_SECOND


def main():
    parser = argparse.ArgumentParser(
        description="Find and embed images from a directory using CLIP"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to search for images (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count images and estimate time, don't actually embed",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories",
    )

    args = parser.parse_args()

    directory = Path(args.directory).resolve()

    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist")
        sys.exit(1)

    if not directory.is_dir():
        print(f"Error: '{directory}' is not a directory")
        sys.exit(1)

    print(f"Scanning: {directory}")
    images = find_images(directory, recursive=not args.no_recursive)

    count = len(images)
    estimated_seconds = count / IMAGES_PER_SECOND if count > 0 else 0

    print(f"Found: {count:,} images")
    print(f"Estimated time: {format_time(estimated_seconds)}")

    if args.dry_run or count == 0:
        return

    # Actually embed the images
    print("\nEmbedding images...")

    image_paths = [str(p) for p in images]

    start = time.perf_counter()

    df = daft.from_pydict({"path": image_paths})
    embed_images = EmbedImages()
    df = df.with_column("embedding", embed_images(col("path")))
    results = df.collect()

    elapsed = time.perf_counter() - start

    print(f"\nDone! Embedded {count:,} images in {format_time(elapsed)}")
    print(f"Speed: {count/elapsed:.1f} images/second")

    # TODO: Save embeddings to disk


if __name__ == "__main__":
    main()
