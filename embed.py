#!/usr/bin/env python3
"""CLI tool to sync image embeddings from a directory."""

import argparse
import os
import sys
import time
from pathlib import Path

import daft
from daft import col

from core import EmbedImages, find_images, format_time, IMAGES_PER_SECOND, DB_PATH


def get_current_files(directory: Path, recursive: bool = True) -> dict[str, float]:
    """Scan directory and return {path: mtime} for all images."""
    images = find_images(directory, recursive=recursive)
    return {str(p): p.stat().st_mtime for p in images}


def get_stored_files() -> dict[str, float]:
    """Read Lance DB and return {path: mtime} for stored embeddings."""
    if not Path(DB_PATH).exists():
        return {}

    df = daft.read_lance(DB_PATH)
    results = df.select("path", "mtime").collect()
    data = results.to_pydict()
    return dict(zip(data["path"], data["mtime"]))


def main():
    parser = argparse.ArgumentParser(
        description="Sync image embeddings from a directory"
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
        help="Show what would be done without actually embedding",
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

    # Scan current files
    print(f"Scanning: {directory}")
    current = get_current_files(directory, recursive=not args.no_recursive)
    print(f"Found: {len(current):,} images")

    # Load stored embeddings
    stored = get_stored_files()
    if stored:
        print(f"Stored: {len(stored):,} embeddings")

    # Compute differences
    current_paths = set(current.keys())
    stored_paths = set(stored.keys())

    new_paths = current_paths - stored_paths
    deleted_paths = stored_paths - current_paths
    common_paths = current_paths & stored_paths

    # Check for modified files (mtime changed)
    modified_paths = {p for p in common_paths if current[p] != stored[p]}
    unchanged_paths = common_paths - modified_paths

    # Paths that need embedding
    to_embed = new_paths | modified_paths

    # Print summary
    print(f"\nUnchanged: {len(unchanged_paths):,}")
    print(f"New: {len(new_paths):,}")
    print(f"Modified: {len(modified_paths):,}")
    print(f"Removed: {len(deleted_paths):,}")

    if not to_embed and not deleted_paths:
        print("\nNothing to do.")
        return

    if to_embed:
        estimated = len(to_embed) / IMAGES_PER_SECOND
        print(f"\nTo embed: {len(to_embed):,} images (~{format_time(estimated)})")

    if args.dry_run:
        return

    # Build the new dataset
    # Keep unchanged embeddings, add new ones, drop deleted
    start = time.perf_counter()

    if to_embed:
        print(f"\nEmbedding {len(to_embed):,} images...")

        # Prepare data for new embeddings
        paths_to_embed = sorted(to_embed)
        mtimes_to_embed = [current[p] for p in paths_to_embed]

        # Create DataFrame and embed
        df_new = daft.from_pydict({"path": paths_to_embed, "mtime": mtimes_to_embed})
        embed_images = EmbedImages()
        df_new = df_new.with_column("vector", embed_images(col("path")))

        # If we have unchanged embeddings, combine them
        if unchanged_paths:
            # Read existing and filter to unchanged only
            df_existing = daft.read_lance(DB_PATH)
            unchanged_list = list(unchanged_paths)
            df_unchanged = df_existing.where(col("path").is_in(unchanged_list))

            # Combine unchanged + new
            df_final = df_unchanged.concat(df_new)
        else:
            df_final = df_new
    else:
        # No new embeddings, just filter out deleted
        df_existing = daft.read_lance(DB_PATH)
        keep_list = list(current_paths)
        df_final = df_existing.where(col("path").is_in(keep_list))

    # Write to Lance
    # Use "create" for new DB, "overwrite" for existing
    mode = "create" if not Path(DB_PATH).exists() else "overwrite"
    df_final.write_lance(DB_PATH, mode=mode)

    elapsed = time.perf_counter() - start

    # Summary
    final_count = len(current) - len(deleted_paths) + len(to_embed) - len(to_embed)  # = len(current)
    print(f"\nDone in {format_time(elapsed)}")
    if to_embed:
        print(f"Speed: {len(to_embed)/elapsed:.1f} images/second")
    print(f"Total embeddings: {len(current):,}")


if __name__ == "__main__":
    main()
