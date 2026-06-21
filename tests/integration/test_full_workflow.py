"""End-to-end workflow tests over a REAL git repo, write tools, pytest
subprocess, and SQLite. Only the LLM-facing planner/coder/reviewer are faked;
everything else is the real thing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agents.results import CodingResult, ReviewResult
from agents.state import PlanningResult
from config import get_settings
from storage.sqlite_store import SqliteRunStore
from tools.file_tools import calculate_sha256
from tools.registry import CODER_TOOL_NAMES, make_tool_executor
from workflow.events import EventType, record_event
from workflow.models import RunStatus
from workflow.rollback import SnapshotStore, make_recording_executor, rollback_run
from workflow.service import WorkflowService

_BASELINE = "def add(a, b):\n    return a + b\n"
_CORRECT = _BASELINE + "\n\ndef subtract(a, b):\n    return a - b\n"
_WRONG = _BASELINE + "\n\ndef subtract(a, b):\n    return a + b\n"

_TEST_FILE = (
    "from calculator import add, subtract\n\n\n"
    "def test_add():\n    assert add(2, 3) == 5\n\n\n"
    "def test_subtract():\n    assert subtract(5, 2) == 3\n"
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


@pytest.fixture()
def live_repo(tmp_path, monkeypatch):
    """A committed calculator repo, with settings pointed at it via env."""
    repo = tmp_path / "calc"
    (repo / "tests").mkdir(parents=True)
    (repo / "calculator.py").write_text(_BASELINE, encoding="utf-8")
    (repo / "tests" / "test_calculator.py").write_text(_TEST_FILE, encoding="utf-8")
    (repo / "conftest.py").write_text("", encoding="utf-8")  # puts root on sys.path
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "T")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "baseline")

    monkeypatch.setenv("TOOL_WORKSPACE_ROOT", str(repo))
    monkeypatch.setenv("DEVAGENT_DATA_DIR", str(repo / ".devagent"))
    monkeypatch.setenv("DEVAGENT_DATABASE_PATH", str(repo / ".devagent" / "devagent.db"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-used-llm-is-faked")
    get_settings.cache_clear()
    yield repo
    get_settings.cache_clear()


def _plan() -> PlanningResult:
    return PlanningResult(
        task="add subtract()", repo_summary="calculator", relevant_files=[],
        implementation_plan=["add a subtract function"],
        files_likely_to_change=["calculator.py"], tests_to_add=[], risks=[],
        unknowns=[], raw_response="{}",
    )


class FakeCoder:
    """Writes calculator.py through the REAL write tools via a recording
    executor, serving a scripted body per attempt."""

    def __init__(self, store, settings, bodies):
        self.store = store
        self.settings = settings
        self.bodies = bodies
        self.calls = 0

    def __call__(self, run, feedback):
        body = self.bodies[min(self.calls, len(self.bodies) - 1)]
        ws = self.settings.tool_workspace_root
        executor = make_recording_executor(
            make_tool_executor(CODER_TOOL_NAMES),
            run_id=str(run.run_id), store=self.store,
            snapshot_store=SnapshotStore(self.settings.devagent_data_dir, str(run.run_id)),
            workspace_root=ws,
        )
        current = (ws / "calculator.py").read_text(encoding="utf-8")
        result = executor(
            "write_file",
            {"path": "calculator.py", "content": body,
             "expected_sha256": calculate_sha256(current)},
        )
        assert result.ok, result.content
        self.calls += 1
        return CodingResult(
            summary="wrote calculator", reported_changed_files=(),
            tests_requested=(), known_issues=(), actual_changed_files=("calculator.py",),
        )


def _review(verdict="approved"):
    return ReviewResult(verdict=verdict, summary="s", issues=(), tests_assessment="p")


def _service(store, settings, coder, reviews):
    review_iter = iter(reviews)
    return WorkflowService(
        store,
        settings=settings,
        planner=lambda task: _plan(),
        coder=coder,
        reviewer=lambda *a: (next(review_iter), 0, 0),
        # verifier left as default => REAL pytest + py_compile subprocess
    )


def test_successful_workflow_end_to_end(live_repo):
    settings = get_settings()
    store = SqliteRunStore(settings.devagent_database_path)
    try:
        svc = _service(store, settings, FakeCoder(store, settings, [_CORRECT]), [_review()])
        run = svc.start_run("add a subtract function")
        assert run.status is RunStatus.AWAITING_PLAN_APPROVAL
        assert (live_repo / "calculator.py").read_text() == _BASELINE  # untouched

        svc.approve_run(str(run.run_id))
        done = svc.resume_run(str(run.run_id))

        assert done.status is RunStatus.COMPLETED
        assert "def subtract" in (live_repo / "calculator.py").read_text()
        assert done.verification_runs[-1].passed  # REAL pytest actually passed
        # Survives a fresh process.
        reloaded = SqliteRunStore(settings.devagent_database_path)
        try:
            assert reloaded.load_run(str(run.run_id)).status is RunStatus.COMPLETED
        finally:
            reloaded.close()
    finally:
        store.close()


def test_repair_after_failing_tests(live_repo):
    settings = get_settings()
    store = SqliteRunStore(settings.devagent_database_path)
    try:
        coder = FakeCoder(store, settings, [_WRONG, _CORRECT])
        svc = _service(store, settings, coder, [_review()])
        run = svc.start_run("add subtract")
        svc.approve_run(str(run.run_id))
        done = svc.resume_run(str(run.run_id))

        assert done.status is RunStatus.COMPLETED
        assert coder.calls == 2  # wrong attempt, then a real repair
        assert len(done.coding_attempts) == 2
    finally:
        store.close()


def test_repair_after_reviewer_requests_changes(live_repo):
    settings = get_settings()
    store = SqliteRunStore(settings.devagent_database_path)
    try:
        # Tests pass both times; the reviewer drives the repair, then approves.
        coder = FakeCoder(store, settings, [_CORRECT, _CORRECT])
        svc = _service(store, settings, coder, [_review("changes_requested"), _review("approved")])
        run = svc.start_run("add subtract")
        svc.approve_run(str(run.run_id))
        done = svc.resume_run(str(run.run_id))

        assert done.status is RunStatus.COMPLETED
        assert coder.calls == 2
    finally:
        store.close()


def test_crash_after_planning_then_resume(live_repo):
    settings = get_settings()
    # Process 1: start and stop.
    store1 = SqliteRunStore(settings.devagent_database_path)
    run = _service(store1, settings, FakeCoder(store1, settings, [_CORRECT]), [_review()]).start_run("add subtract")
    run_id = str(run.run_id)
    store1.close()

    # Process 2: a brand-new store + service approve and finish.
    store2 = SqliteRunStore(settings.devagent_database_path)
    try:
        svc = _service(store2, settings, FakeCoder(store2, settings, [_CORRECT]), [_review()])
        svc.approve_run(run_id)
        done = svc.resume_run(run_id)
        assert done.status is RunStatus.COMPLETED
        assert "def subtract" in (live_repo / "calculator.py").read_text()
    finally:
        store2.close()


def test_rollback_restores_baseline(live_repo):
    settings = get_settings()
    store = SqliteRunStore(settings.devagent_database_path)
    try:
        svc = _service(store, settings, FakeCoder(store, settings, [_CORRECT]), [_review()])
        run = svc.start_run("add subtract")
        svc.approve_run(str(run.run_id))
        svc.resume_run(str(run.run_id))
        assert "def subtract" in (live_repo / "calculator.py").read_text()

        rollback_run(str(run.run_id), store, settings)
        assert (live_repo / "calculator.py").read_text() == _BASELINE
        assert store.load_run(str(run.run_id)).status is RunStatus.ROLLED_BACK
    finally:
        store.close()


def test_event_payload_is_redacted_in_the_database(live_repo):
    settings = get_settings()
    store = SqliteRunStore(settings.devagent_database_path)
    try:
        svc = _service(store, settings, FakeCoder(store, settings, [_CORRECT]), [_review()])
        run = svc.start_run("add subtract")
        record_event(
            store, str(run.run_id), EventType.TOOL_COMPLETED,
            payload={"leak": "key is sk-ant-supersecret123"},
        )
        # Reload from the db and confirm the secret never landed.
        stored = SqliteRunStore(settings.devagent_database_path)
        try:
            payloads = [e.payload for e in stored.list_events(str(run.run_id))]
        finally:
            stored.close()
        assert any("[REDACTED]" in str(p) for p in payloads)
        assert all("sk-ant-supersecret123" not in str(p) for p in payloads)
    finally:
        store.close()
