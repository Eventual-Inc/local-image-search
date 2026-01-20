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
        "REFRESH_INTERVAL": "60"
      }
    }
  }
}
```

- **command** - The executable to run (`uvx` for Python packages)
- **args** - CLI arguments (first is package name, rest are passed to the server)
- **env** - Environment variables (isolated from shell, must be explicit)

### Implementation Plan

1. ~~Add `mcp` SDK to dependencies~~ Done
2. ~~Create MCP server that exposes `search_images` tool~~ Done
3. ~~Add console script entry point in pyproject.toml~~ Done
4. Auto-download model if not present (handle missing model gracefully)
5. ~~Background thread to refresh embeddings~~ Done
   - Runs immediately on startup (initial sync)
   - Default: every 60 seconds (configurable via `REFRESH_INTERVAL` env var)
   - Separate thread so it doesn't block MCP requests
6. ~~Take image directory as CLI argument~~ Done
7. Users run: `uvx local-image-search ~/Pictures`

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REFRESH_INTERVAL` | `60` | Seconds between embedding refresh cycles |

Example with custom interval:
```json
{
  "mcpServers": {
    "local-image-search": {
      "command": "uvx",
      "args": ["local-image-search", "/Users/username/Pictures"],
      "env": {
        "REFRESH_INTERVAL": "30"
      }
    }
  }
}
```

### MCP Server Setup (Development)

To test locally during development:

```bash
# Add to Claude Code (must split command and args properly)
claude mcp add -s user local-image-search -- uv --directory /Users/yk/Desktop/projects/local-image-search run python mcp_server.py

# Restart Claude Code to load the server
```

**Gotchas we encountered:**

1. **stdout is reserved for MCP protocol** - Any `print()` statements corrupt the JSON-RPC communication. Use `print(..., file=sys.stderr)` for logging.

2. **Command must be split from args** - This is wrong:
   ```json
   "command": "uv --directory /path run python mcp_server.py",
   "args": []
   ```
   This is correct:
   ```json
   "command": "uv",
   "args": ["--directory", "/path", "run", "python", "mcp_server.py"]
   ```

3. **Restart required** - Claude Code must be fully restarted (not just new conversation) to pick up MCP config changes.

