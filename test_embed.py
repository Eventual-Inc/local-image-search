#!/usr/bin/env python3
"""Tests for embed.py sync logic."""

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

# Path to the Lance DB (will be deleted at start of each test run)
DB_PATH = Path(__file__).parent / "embeddings.lance"

# Source images for testing
POKEMON_DIR = Path(__file__).parent / "data" / "pokemon"


def run_embed(directory: str, dry_run: bool = False) -> str:
    """Run embed.py and return output."""
    cmd = ["uv", "run", "python", "embed.py", directory]
    if dry_run:
        cmd.append("--dry-run")
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout + result.stderr


def parse_output(output: str) -> dict:
    """Parse embed.py output into a dict of counts."""
    counts = {}
    for line in output.split("\n"):
        if ": " in line:
            parts = line.split(": ")
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip().replace(",", "")
                # Try to parse as int
                try:
                    counts[key] = int(value.split()[0])
                except (ValueError, IndexError):
                    counts[key] = value
    return counts


def reset_db():
    """Delete the Lance DB to start fresh."""
    if DB_PATH.exists():
        shutil.rmtree(DB_PATH)
    print("DB reset.")


def test_new_images():
    """Test: New images should be embedded."""
    print("\n=== Test: New Images ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Copy 5 images
        for i, img in enumerate(sorted(POKEMON_DIR.glob("*.png"))[:5]):
            shutil.copy(img, tmpdir / img.name)

        # Run embed
        output = run_embed(str(tmpdir))
        counts = parse_output(output)
        print(output)

        assert counts.get("Found") == 5, f"Expected 5 found, got {counts.get('Found')}"
        assert counts.get("New") == 5, f"Expected 5 new, got {counts.get('New')}"
        assert counts.get("Unchanged") == 0, f"Expected 0 unchanged, got {counts.get('Unchanged')}"

        print("PASSED: 5 new images embedded")


def test_unchanged_images():
    """Test: Running again should skip all images."""
    print("\n=== Test: Unchanged Images ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Copy 5 images
        for img in sorted(POKEMON_DIR.glob("*.png"))[:5]:
            shutil.copy(img, tmpdir / img.name)

        # First run - embed all
        run_embed(str(tmpdir))

        # Second run - should skip all
        output = run_embed(str(tmpdir))
        counts = parse_output(output)
        print(output)

        assert counts.get("Unchanged") == 5, f"Expected 5 unchanged, got {counts.get('Unchanged')}"
        assert counts.get("New") == 0, f"Expected 0 new, got {counts.get('New')}"
        assert "Nothing to do" in output, "Expected 'Nothing to do' message"

        print("PASSED: All 5 images skipped (cached)")


def test_modified_image():
    """Test: Modified image should be re-embedded."""
    print("\n=== Test: Modified Image ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Copy 5 images
        images = sorted(POKEMON_DIR.glob("*.png"))[:5]
        for img in images:
            shutil.copy(img, tmpdir / img.name)

        # First run - embed all
        run_embed(str(tmpdir))

        # Modify one image (change mtime)
        time.sleep(0.1)  # Ensure mtime changes
        target = tmpdir / images[0].name
        target.touch()

        # Second run - should re-embed 1
        output = run_embed(str(tmpdir))
        counts = parse_output(output)
        print(output)

        assert counts.get("Unchanged") == 4, f"Expected 4 unchanged, got {counts.get('Unchanged')}"
        assert counts.get("Modified") == 1, f"Expected 1 modified, got {counts.get('Modified')}"

        print("PASSED: 1 modified image re-embedded")


def test_deleted_image():
    """Test: Deleted image should be removed from DB."""
    print("\n=== Test: Deleted Image ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Copy 5 images
        images = sorted(POKEMON_DIR.glob("*.png"))[:5]
        for img in images:
            shutil.copy(img, tmpdir / img.name)

        # First run - embed all
        run_embed(str(tmpdir))

        # Delete one image
        (tmpdir / images[0].name).unlink()

        # Second run - should remove 1
        output = run_embed(str(tmpdir))
        counts = parse_output(output)
        print(output)

        assert counts.get("Found") == 4, f"Expected 4 found, got {counts.get('Found')}"
        assert counts.get("Removed") == 1, f"Expected 1 removed, got {counts.get('Removed')}"
        assert counts.get("Unchanged") == 4, f"Expected 4 unchanged, got {counts.get('Unchanged')}"

        print("PASSED: 1 deleted image removed from DB")


def test_added_after_initial():
    """Test: Adding new images after initial embed."""
    print("\n=== Test: Add Images After Initial ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Copy 3 images initially
        images = sorted(POKEMON_DIR.glob("*.png"))[:5]
        for img in images[:3]:
            shutil.copy(img, tmpdir / img.name)

        # First run - embed 3
        run_embed(str(tmpdir))

        # Add 2 more images
        for img in images[3:5]:
            shutil.copy(img, tmpdir / img.name)

        # Second run - should embed 2 new, keep 3
        output = run_embed(str(tmpdir))
        counts = parse_output(output)
        print(output)

        assert counts.get("Found") == 5, f"Expected 5 found, got {counts.get('Found')}"
        assert counts.get("Unchanged") == 3, f"Expected 3 unchanged, got {counts.get('Unchanged')}"
        assert counts.get("New") == 2, f"Expected 2 new, got {counts.get('New')}"

        print("PASSED: 2 new images added, 3 unchanged")


def test_mixed_changes():
    """Test: Mix of new, modified, deleted, unchanged."""
    print("\n=== Test: Mixed Changes ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Copy 5 images initially
        images = sorted(POKEMON_DIR.glob("*.png"))[:7]
        for img in images[:5]:
            shutil.copy(img, tmpdir / img.name)

        # First run - embed 5
        run_embed(str(tmpdir))

        time.sleep(0.1)

        # Make changes:
        # - Delete 1 (index 0)
        (tmpdir / images[0].name).unlink()
        # - Modify 1 (index 1)
        (tmpdir / images[1].name).touch()
        # - Add 2 new (index 5, 6)
        for img in images[5:7]:
            shutil.copy(img, tmpdir / img.name)
        # - Keep 3 unchanged (index 2, 3, 4)

        # Second run
        output = run_embed(str(tmpdir))
        counts = parse_output(output)
        print(output)

        assert counts.get("Found") == 6, f"Expected 6 found, got {counts.get('Found')}"
        assert counts.get("Unchanged") == 3, f"Expected 3 unchanged, got {counts.get('Unchanged')}"
        assert counts.get("New") == 2, f"Expected 2 new, got {counts.get('New')}"
        assert counts.get("Modified") == 1, f"Expected 1 modified, got {counts.get('Modified')}"
        assert counts.get("Removed") == 1, f"Expected 1 removed, got {counts.get('Removed')}"

        print("PASSED: Mixed changes handled correctly")


def main():
    print("Starting embed.py tests...")
    print(f"Pokemon images: {POKEMON_DIR}")
    print(f"DB path: {DB_PATH}")

    # Reset DB before all tests
    reset_db()

    # Run tests (reset DB between each to isolate)
    tests = [
        test_new_images,
        test_unchanged_images,
        test_modified_image,
        test_deleted_image,
        test_added_after_initial,
        test_mixed_changes,
    ]

    passed = 0
    failed = 0

    for test in tests:
        reset_db()
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")

    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
