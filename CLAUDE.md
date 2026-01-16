# Local Image Search - Project Notes

See README.md for project structure and setup instructions.

## What We've Done

- Set up MLX CLIP from ml-explore/mlx-examples
- Created Daft-based batch embedding with `@daft.cls`
- Benchmarked performance (see benchmark_plot.png)
- Added Pokemon dataset (1025 images) for testing

## Next Steps

1. Embed all images on laptop locally
2. Figure out storage format for embeddings
3. Avoid re-embedding the same image, even if filename changed
   - Key insight: we want **image content → embedding**, not **filename → embedding**
4. Create `setup.sh` script that combines `uv sync` and model download
5. Fix paths so embedding function works from anywhere (not just project root)
