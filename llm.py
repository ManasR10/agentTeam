from __future__ import annotations

from dataclasses import dataclass

from anthropic import Anthropic  # pyright: ignore[reportMissingImports]

from config import settings


@dataclass(frozen=True, slots=True)
class LLMResult:
    """Normalized result returned by an NPW text-model call."""

    text: str
    model: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int


class EmptyLLMResponseError(RuntimeError):
    """Raised when the model response contains no text."""


client = Anthropic(
    api_key=settings.anthropic_api_key,
    timeout=settings.llm_timeout_seconds,
)


def call_llm(
    prompt: str,
    *,
    system: str = "You are a helpful software engineering assistant.",
    max_tokens: int | None = None,
) -> LLMResult:
    """
    Make a single-turn, text-only Claude API request.

    This function intentionally does not implement tools, conversation
    history, streaming, retries, or LangGraph orchestration. Those belong
    to later NPW phases.

    Args:
        prompt:
            User instruction sent to Claude.
        system:
            System instruction defining the model's behaviour.
        max_tokens:
            Optional per-call output token limit. Uses the configured
            default when omitted.

    Returns:
        LLMResult containing text and basic usage metadata.

    Raises:
        ValueError:
            If prompt or system is empty.
        EmptyLLMResponseError:
            If the response does not contain a text block.
        anthropic.APIError:
            For API, connection, authentication, or rate-limit errors.
    """
    clean_prompt = prompt.strip()
    clean_system = system.strip()
    if not clean_prompt:
        raise ValueError("prompt cannot be empty")
    if not clean_system:
        raise ValueError("system cannot be empty")

    resolved_max_tokens = max_tokens or settings.llm_max_tokens
    if resolved_max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")

    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=resolved_max_tokens,
        system=clean_system,
        messages=[
            {
                "role": "user",
                "content": clean_prompt,
            }
        ],
    )

    text_blocks = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    ]
    text = "".join(text_blocks).strip()

    if not text:
        block_types = [
            getattr(block, "type", "unknown")
            for block in response.content
        ]
        raise EmptyLLMResponseError(
            "Claude returned no text content. "
            f"Received block types: {block_types}. "
            f"Stop reason: {response.stop_reason!r}."
        )

    return LLMResult(
        text=text,
        model=response.model,
        stop_reason=response.stop_reason,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


def call_llm_text(
    prompt: str,
    *,
    system: str = "You are a helpful software engineering assistant.",
    max_tokens: int | None = None,
) -> str:
    """Convenience wrapper for callers that only need generated text."""
    result = call_llm(
        prompt,
        system=system,
        max_tokens=max_tokens,
    )
    return result.text
