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
        max_files_changed=10,
        max_file_write_chars=100000,
        max_total_write_chars=300000,
        allow_file_creation=True,
        allow_file_overwrite=True,
        command_timeout_seconds=120,
        max_command_output_chars=30000,
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

    # Inject a fake executor so the loop never touches the real filesystem.
    def fake_executor(name, inp):
        return ToolResult(
            ok=True, content="MOCK CONTENTS", metadata={"path": inp.get("path")}
        )

    result = llm.call_agent_with_tools(
        "read the readme", tool_executor=fake_executor
    )

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


def test_default_profile_is_read_only(monkeypatch, tmp_path) -> None:
    """A plain caller (no profile) must never be advertised write/command tools."""
    client = _FakeClient([_Response([_TextBlock("done")], "end_turn")])
    _patch(monkeypatch, _settings(tmp_path), client)

    llm.call_agent_with_tools("hi")

    advertised = {t["name"] for t in client.messages.create_calls[0]["tools"]}
    assert advertised == {"read_file", "list_files"}
    assert "write_file" not in advertised
    assert "run_tests" not in advertised


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

    with pytest.raises(llm.ToolLoopError):
        llm.call_agent_with_tools(
            "loop forever",
            tool_executor=lambda name, inp: ToolResult(
                ok=True, content="loop", metadata=None
            ),
        )

    # the loop stops exactly at the iteration cap
    assert len(client.messages.create_calls) == settings.tool_max_iterations
