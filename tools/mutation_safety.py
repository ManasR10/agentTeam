from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import Settings

# Mutation safety is intentionally stricter than the read-side ignore list in
# file_tools. Reading is about hiding noise; mutation is about preventing harm.


@dataclass(frozen=True, slots=True)
class MutationPolicy:
    """Bounds on how much a single agent run may write, built from Settings."""

    max_files_changed: int
    max_file_write_chars: int
    max_total_write_chars: int
    allow_create_files: bool
    allow_overwrite_files: bool


def build_mutation_policy(settings: Settings) -> MutationPolicy:
    """Project the relevant Settings fields into a MutationPolicy view."""
    return MutationPolicy(
        max_files_changed=settings.max_files_changed,
        max_file_write_chars=settings.max_file_write_chars,
        max_total_write_chars=settings.max_total_write_chars,
        allow_create_files=settings.allow_file_creation,
        allow_overwrite_files=settings.allow_file_overwrite,
    )


# Directory names that must never be written into, anywhere in the path.
PROTECTED_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        "dist",
        "build",
        "coverage",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
)

# Exact file names that must never be modified, even though they are text.
# Lockfiles are blocked until controlled package-manager support exists; we do
# not let the agent edit dependency pins by hand. .env.example is readable but
# not auto-modifiable during Phase 3.
PROTECTED_FILE_NAMES = frozenset(
    {
        ".env.example",
        "requirements.lock.txt",
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
    }
)

# Suffixes for secrets and binary stores that must never be written.
PROTECTED_SUFFIXES = frozenset(
    {
        ".pem",
        ".key",
        ".crt",
        ".p12",
        ".sqlite",
        ".db",
    }
)


def is_protected_mutation_path(relative_path: Path) -> bool:
    """
    Return True if `relative_path` (relative to the workspace) must not be
    written, created, or overwritten by an agent tool.

    This blocks any `.env*` secret (except is handled by the caller's read
    rules — here every `.env*` including `.env.example` is non-writable),
    protected directories at any depth, lockfiles, and secret/binary suffixes.
    """
    name = relative_path.name

    # Any .env* file, including .env.example, is non-writable in Phase 3.
    if name == ".env" or name.startswith(".env"):
        return True
    if name in PROTECTED_FILE_NAMES:
        return True
    if relative_path.suffix.lower() in PROTECTED_SUFFIXES:
        return True
    if set(relative_path.parts).intersection(PROTECTED_DIR_NAMES):
        return True
    return False
