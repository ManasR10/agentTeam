from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE)


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is invalid."""


def read_positive_integer(name: str, default: int) -> int:
    """
    Read a positive integer environment variable.

    Raises:
        ConfigurationError: If the value is not a positive integer.
    """
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(
            f"{name} must be an integer, received: {raw_value!r}"
        ) from exc
    if value <= 0:
        raise ConfigurationError(
            f"{name} must be greater than zero, received: {value}"
        )
    return value


def resolve_workspace_root(raw_path: str) -> Path:
    """
    Resolve TOOL_WORKSPACE_ROOT into an absolute directory path.

    A relative value is resolved against PROJECT_ROOT so the workspace
    is stable no matter which directory the program is launched from.

    Raises:
        ConfigurationError: If the path does not exist or is not a directory.
    """
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    if not resolved.exists():
        raise ConfigurationError(
            f"TOOL_WORKSPACE_ROOT does not exist: {resolved}"
        )
    if not resolved.is_dir():
        raise ConfigurationError(
            f"TOOL_WORKSPACE_ROOT must be a directory: {resolved}"
        )
    return resolved


@dataclass(frozen=True, slots=True)
class Settings:
    # Optional at load time: the key is only required to build the Anthropic
    # client (see llm.get_client). Keeping it optional lets offline tools and
    # tests load settings without a key present.
    anthropic_api_key: str | None
    llm_model: str
    llm_max_tokens: int
    llm_timeout_seconds: int
    tool_max_iterations: int
    tool_workspace_root: Path
    max_file_read_chars: int


def load_settings() -> Settings:
    """Load and validate the DevAgent application settings."""
    model = os.getenv("LLM_MODEL", "claude-sonnet-4-6").strip()
    if not model:
        raise ConfigurationError("LLM_MODEL cannot be empty.")
    workspace_root = resolve_workspace_root(
        os.getenv("TOOL_WORKSPACE_ROOT", ".")
    )
    raw_api_key = os.getenv("ANTHROPIC_API_KEY")
    api_key = raw_api_key.strip() if raw_api_key and raw_api_key.strip() else None
    return Settings(
        anthropic_api_key=api_key,
        llm_model=model,
        llm_max_tokens=read_positive_integer(
            "LLM_MAX_TOKENS",
            default=1024,
        ),
        llm_timeout_seconds=read_positive_integer(
            "LLM_TIMEOUT_SECONDS",
            default=60,
        ),
        tool_max_iterations=read_positive_integer(
            "TOOL_MAX_ITERATIONS",
            default=5,
        ),
        tool_workspace_root=workspace_root,
        max_file_read_chars=read_positive_integer(
            "MAX_FILE_READ_CHARS",
            default=20000,
        ),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the validated application settings.

    Settings are loaded and validated on first call (not at import time) and
    then cached. Loading lazily means importing this module can never fail
    just because `.env` is missing, which keeps tests and tooling importable.
    """
    return load_settings()
