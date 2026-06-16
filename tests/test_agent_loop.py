from __future__ import annotations

from pathlib import Path

import pytest

import llm
from config import Settings
from tools.schemas import ToolResult


# --------------------------------------------------------------------------
# Tiny fakes that mimic just enough of the Anthropic SDK response shape.
# --------------------------------------------------------------------------
class _Usage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _TextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _ToolUseBlock:
    type = "tool_use"

    def __init__(self, id: str, name: str, input: dict) -> None:
        self.id = id
        self.name = name
        self.input = input


class _Response:
    def __init__(self, content: list, stop_reason: str) -> None:
        self.content = content
        self.model = "fake-model"
        self.stop_reason = stop_reason
        self.usage = _Usage(11, 7)


class _Messages:
    def __init__(self, scripted: list[_Response]) -> None:
        self._scripted = list(scripted)
        self.create_calls: list[dict] = []

    def create(self, **kwargs):
        # Snapshot the messages list, which the caller keeps mutating, so each
        # recorded call reflects the conversation as it was at that moment.
        snapshot = dict(kwargs)
        if "messages" in snapshot:
            snapshot["messages"] = list(snapshot["messages"])
        self.create_calls.append(snapshot)
        return self._scripted.pop(0)


class _FakeClient:
    def __init__(self, scripted: list[_Response]) -> None:
        self.messages = _Messages(scripted)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        anthropic_api_key="test-key",
        llm_model="fake-model",
        llm_max_tokens=256,
        llm_timeout_seconds=30,
        tool_max_iterations=3,
        tool_workspace_root=tmp_path,
        max_file_read_chars=1000,
    )


def _patch(monkeypatch, settings: Settings, client: _FakeClient) -> None:
    monkeypatch.setattr(llm, "get_settings", lambda: settings)
    monkeypatch.setattr(llm, "get_client", lambda: client)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_returns_immediately_when_no_tool_used(monkeypatch, tmp_path) -> None:
    client = _FakeClient(
        [_Response([_TextBlock("Direct answer.")], "end_turn")]
    )
    _patch(monkeypatch, _settings(tmp_path), client)

    result = llm.call_agent_with_tools("hi")

    assert result.text == "Direct answer."
    assert result.iterations == 1
    assert result.tool_calls == []
    assert result.input_tokens == 11
    assert result.output_tokens == 7


def test_executes_tool_then_returns_final_text(monkeypatch, tmp_path) -> None:
    client = _FakeClient(
        [
            _Response(
                [_ToolUseBlock("toolu_1", "read_file", {"path": "README.md"})],
                "tool_use",
            ),
            _Response([_TextBlock("Based on the file, X.")], "end_turn"),
        ]
    )
    _patch(monkeypatch, _settings(tmp_path), client)
    # Mock the dispatcher so the loop never touches the real filesystem.
    monkeypatch.setattr(
        llm,
        "execute_tool",
        lambda name, inp: ToolResult(
            ok=True, content="MOCK CONTENTS", metadata={"path": inp.get("path")}
        ),
    )

    result = llm.call_agent_with_tools("read the readme")

    assert result.text == "Based on the file, X."
    assert result.iterations == 2
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.name == "read_file"
    assert call.ok is True
    assert call.input == {"path": "README.md"}
    # tokens accumulate across both API calls
    assert result.input_tokens == 22
    assert result.output_tokens == 14
    # the second API call must send the tool_result back, linked by id
    second_messages = client.messages.create_calls[1]["messages"]
    last = second_messages[-1]
    assert last["role"] == "user"
    assert last["content"][0]["type"] == "tool_result"
    assert last["content"][0]["tool_use_id"] == "toolu_1"


def test_raises_after_max_iterations(monkeypatch, tmp_path) -> None:
    settings = _settings(tmp_path)  # tool_max_iterations == 3
    always_tool = [
        _Response(
            [_ToolUseBlock(f"toolu_{i}", "read_file", {"path": "x"})],
            "tool_use",
        )
        for i in range(10)
    ]
    client = _FakeClient(always_tool)
    _patch(monkeypatch, settings, client)
    monkeypatch.setattr(
        llm,
        "execute_tool",
        lambda name, inp: ToolResult(ok=True, content="loop", metadata=None),
    )

    with pytest.raises(llm.ToolLoopError):
        llm.call_agent_with_tools("loop forever")

    # the loop stops exactly at the iteration cap
    assert len(client.messages.create_calls) == settings.tool_max_iterations
