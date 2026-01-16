#!/usr/bin/env python3
"""CLI tool to search images via the server."""

import argparse
import subprocess
import sys

import requests


def main():
    parser = argparse.ArgumentParser(description="Search for images")
    parser.add_argument("query", help="Search query")
    parser.add_argument("-n", "--limit", type=int, default=5, help="Number of results")
    parser.add_argument("-o", "--open", action="store_true", help="Open results in default viewer")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Server URL")
    args = parser.parse_args()

    try:
        response = requests.post(
            f"{args.url}/search",
            json={"query": args.query, "limit": args.limit},
            timeout=10,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print("Error: Server not running. Start with: uv run python server.py")
        sys.exit(1)

    data = response.json()
    results = data["results"]

    if not results:
        print("No results found.")
        return

    print(f"Found {data['total_images']} images, showing top {len(results)}:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['score']:.3f}] {r['path']}")

    if args.open and results:
        paths = [r["path"] for r in results]
        subprocess.run(["open"] + paths)


if __name__ == "__main__":
    main()
