from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from anthropic import Anthropic  # pyright: ignore[reportMissingImports]

from config import ConfigurationError, get_settings
from tools.registry import (
    READ_ONLY_TOOL_NAMES,
    get_tool_definitions,
    make_tool_executor,
)
from tools.schemas import ToolResult

# A tool executor maps (tool_name, tool_input) to a ToolResult. The loop stays
# agnostic about which tools are permitted; the executor closes over that.
ToolExecutor = Callable[[str, dict[str, Any]], ToolResult]


@dataclass(frozen=True, slots=True)
class LLMResult:
    """Normalized result returned by a DevAgent text-only model call."""

    text: str
    model: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    """Trace record for one tool call made during a model run."""

    name: str
    input: dict[str, Any]
    ok: bool
    content_preview: str


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """
    Final result of a manual tool-use run.

    This is not LangGraph yet. It is only a single manual loop around
    Anthropic's Messages API.
    """

    text: str
    model: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    tool_calls: list[ToolCallRecord]
    iterations: int


class EmptyLLMResponseError(RuntimeError):
    """Raised when the model response contains no text."""


class ToolLoopError(RuntimeError):
    """Raised when the manual tool loop cannot complete safely."""


@lru_cache(maxsize=1)
def get_client() -> Anthropic:
    """
    Return the cached Anthropic client.

    Created on first call (not at import time) so importing this module does
    not require a valid API key — important for tests and tooling. The key is
    validated here, at the only point that actually needs it, rather than when
    settings load (which offline file tools also trigger).
    """
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise ConfigurationError(
            "Missing required environment variable: ANTHROPIC_API_KEY. "
            "Create a .env file using .env.example as the template."
        )
    return Anthropic(
        api_key=settings.anthropic_api_key,
        timeout=settings.llm_timeout_seconds,
    )


def extract_text_from_blocks(content_blocks: list[Any]) -> str:
    """Join the text of all `text` blocks in a response, ignoring others."""
    text_parts: list[str] = []
    for block in content_blocks:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)
    return "".join(text_parts).strip()


def call_llm(
    prompt: str,
    *,
    system: str = "You are a helpful software engineering assistant.",
    max_tokens: int | None = None,
) -> LLMResult:
    """
    Phase 0 text-only call.

    Kept for simple direct calls that do not need tools.
    """
    clean_prompt = prompt.strip()
    clean_system = system.strip()
    if not clean_prompt:
        raise ValueError("prompt cannot be empty")
    if not clean_system:
        raise ValueError("system cannot be empty")

    settings = get_settings()
    resolved_max_tokens = (
        max_tokens if max_tokens is not None else settings.llm_max_tokens
    )
    if resolved_max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")

    response = get_client().messages.create(
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

    text = extract_text_from_blocks(response.content)
    if not text:
        raise EmptyLLMResponseError(
            f"Claude returned no text. Stop reason: {response.stop_reason!r}"
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


def format_tool_result_content(
    *,
    tool_name: str,
    ok: bool,
    content: str,
    metadata: dict[str, Any] | None,
) -> str:
    """
    Format tool output as a compact JSON string for Claude.

    Returning JSON makes tool results easy for the model to reason about.
    """
    payload = {
        "tool_name": tool_name,
        "ok": ok,
        "content": content,
        "metadata": metadata or {},
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def call_agent_with_tools(
    prompt: str,
    *,
    system: str = (
        "You are DevAgent, a careful software-engineering assistant. "
        "Use tools when you need to inspect project files. "
        "Do not guess file contents. "
        "If a tool fails, explain the failure clearly."
    ),
    tools: Sequence[dict[str, Any]] | None = None,
    tool_executor: ToolExecutor | None = None,
    max_tokens: int | None = None,
    max_iterations: int | None = None,
) -> AgentRunResult:
    """
    Run a manual Anthropic tool-use loop.

    This function:
    1. Sends the user prompt and tool schemas to Claude.
    2. Detects tool_use blocks.
    3. Executes requested tools in Python.
    4. Sends tool_result blocks back to Claude.
    5. Repeats until Claude returns final text or the iteration cap is hit.

    This is the core of Phase 1.
    """
    clean_prompt = prompt.strip()
    clean_system = system.strip()
    if not clean_prompt:
        raise ValueError("prompt cannot be empty")
    if not clean_system:
        raise ValueError("system cannot be empty")

    # SECURE DEFAULT: with no profile, only read-only tools are advertised.
    # Now that the registry also holds write/command/git tools, defaulting to
    # the full set would hand any plain caller (e.g. phase1_demo) mutation
    # power. Callers that need more must opt in explicitly.
    if tools is None:
        resolved_tools = get_tool_definitions(READ_ONLY_TOOL_NAMES)
    else:
        resolved_tools = list(tools)
    # If no executor is supplied, permit exactly the advertised tools, so the
    # allow-list can never be wider than what the model was shown.
    if tool_executor is None:
        advertised = {tool["name"] for tool in resolved_tools}
        resolved_executor: ToolExecutor = make_tool_executor(advertised)
    else:
        resolved_executor = tool_executor

    settings = get_settings()
    resolved_max_tokens = (
        max_tokens if max_tokens is not None else settings.llm_max_tokens
    )
    if resolved_max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")

    resolved_max_iterations = (
        max_iterations
        if max_iterations is not None
        else settings.tool_max_iterations
    )
    if resolved_max_iterations <= 0:
        raise ValueError("max_iterations must be greater than zero")

    # The running conversation. We append to this every turn: the assistant's
    # tool_use request, then our tool_result reply, then the assistant again.
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": clean_prompt,
        }
    ]

    tool_call_records: list[ToolCallRecord] = []
    total_input_tokens = 0
    total_output_tokens = 0
    last_model = settings.llm_model
    last_stop_reason: str | None = None

    client = get_client()
    for iteration in range(1, resolved_max_iterations + 1):
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=resolved_max_tokens,
            system=clean_system,
            tools=resolved_tools,
            messages=messages,
        )

        last_model = response.model
        last_stop_reason = response.stop_reason
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        # The assistant turn (which may contain tool_use blocks) must be added
        # to history before we send tool results back.
        messages.append(
            {
                "role": "assistant",
                "content": response.content,
            }
        )

        tool_use_blocks = [
            block
            for block in response.content
            if getattr(block, "type", None) == "tool_use"
        ]

        # No tool requested -> Claude has given its final answer.
        if not tool_use_blocks:
            text = extract_text_from_blocks(response.content)
            if not text:
                raise EmptyLLMResponseError(
                    "Claude returned neither text nor tool_use blocks. "
                    f"Stop reason: {response.stop_reason!r}"
                )
            return AgentRunResult(
                text=text,
                model=last_model,
                stop_reason=last_stop_reason,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                tool_calls=tool_call_records,
                iterations=iteration,
            )

        # Execute every requested tool and collect tool_result blocks.
        tool_result_blocks: list[dict[str, Any]] = []
        for tool_use in tool_use_blocks:
            tool_name = tool_use.name
            tool_input = tool_use.input

            if not isinstance(tool_input, dict):
                tool_result_text = format_tool_result_content(
                    tool_name=tool_name,
                    ok=False,
                    content="Tool input must be a JSON object.",
                    metadata={"received_input": repr(tool_input)},
                )
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": tool_result_text,
                        "is_error": True,
                    }
                )
                tool_call_records.append(
                    ToolCallRecord(
                        name=tool_name,
                        input={},
                        ok=False,
                        content_preview="Tool input must be a JSON object.",
                    )
                )
                continue

            result = resolved_executor(tool_name, tool_input)

            tool_result_text = format_tool_result_content(
                tool_name=tool_name,
                ok=result.ok,
                content=result.content,
                metadata=result.metadata,
            )
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": tool_result_text,
                    "is_error": not result.ok,
                }
            )

            preview = result.content[:300]
            if len(result.content) > 300:
                preview += "..."
            tool_call_records.append(
                ToolCallRecord(
                    name=tool_name,
                    input=tool_input,
                    ok=result.ok,
                    content_preview=preview,
                )
            )

        # Send all tool results back as a single user turn.
        messages.append(
            {
                "role": "user",
                "content": tool_result_blocks,
            }
        )

    raise ToolLoopError(
        "Tool loop reached maximum iterations "
        f"({resolved_max_iterations}) without final response."
    )
