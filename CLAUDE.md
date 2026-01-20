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

## MCP Server Plan

Goal: Make this an MCP server users can install with a single command.

### How MCP Configuration Works

Claude Desktop/Code config (`claude_desktop_config.json` or `.claude.json`):
```json
{
  "mcpServers": {
    "local-image-search": {
      "command": "uvx",
      "args": ["local-image-search", "/Users/username/Pictures"],
      "env": {
        "SOME_VAR": "value"
      }
    }
  }
}
```

- **command** - The executable to run (`uvx` for Python packages)
- **args** - CLI arguments (first is package name, rest are passed to the server)
- **env** - Environment variables (isolated from shell, must be explicit)

### Implementation Plan

1. Add `mcp` SDK to dependencies
2. Create MCP server that exposes `search_images` tool
3. Take image directory as CLI argument
4. Make pip-installable with console script entry point
5. Users run: `uvx local-image-search ~/Pictures`

