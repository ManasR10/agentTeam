from __future__ import annotations

# Shared test data for the Phase 2 planner/formatting/demo tests. Kept in one
# place so test modules don't import constants from each other.
VALID_PLANNER_JSON = """
{
  "task": "Add CLI",
  "repo_summary": "DevAgent has an LLM wrapper and read-only tools.",
  "relevant_files": [
    {
      "path": "llm.py",
      "reason": "Contains the tool loop."
    },
    {
      "path": "README.md",
      "reason": "Needs usage documentation."
    }
  ],
  "implementation_plan": [
    "Create cli.py",
    "Wire cli.py to the planning agent",
    "Document the command"
  ],
  "files_likely_to_change": [
    "cli.py",
    "README.md"
  ],
  "tests_to_add": [
    "test_cli_default_task",
    "test_cli_custom_task"
  ],
  "risks": [
    "Avoid live API calls in unit tests"
  ],
  "unknowns": [
    "Exact CLI command name is not decided"
  ]
}
""".strip()
