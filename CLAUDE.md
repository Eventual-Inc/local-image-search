# Local Image Search - Project Notes

See README.md for project structure and setup instructions.

## What We've Done

- Set up MLX CLIP from ml-explore/mlx-examples
- Created Daft-based batch embedding with `@daft.cls`
- Benchmarked performance (see benchmark_plot.png)
- Added Pokemon dataset (1025 images) for testing
- Implemented Lance DB storage for embeddings
- Added FastAPI server and search CLI
- Added incremental embedding (skips unchanged files by path + mtime)
- Added error handling for corrupted/unreadable images

