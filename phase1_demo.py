from __future__ import annotations

from llm import call_agent_with_tools


def main() -> int:
    task = (
        "List the files in the project root, then read README.md, "
        "then explain in 5 bullet points what this project currently does."
    )

    print("Running DevAgent Phase 1 tool-use demo...")
    print()
    print("Task:")
    print(task)
    print()

    try:
        result = call_agent_with_tools(task)
    except Exception as exc:
        print("Phase 1 demo failed.")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error: {exc}")
        return 1

    print("Final answer:")
    print(result.text)
    print()

    print("Tool calls:")
    if not result.tool_calls:
        print("- No tools used")
    else:
        for index, tool_call in enumerate(result.tool_calls, start=1):
            print(f"{index}. {tool_call.name}")
            print(f"   input: {tool_call.input}")
            print(f"   ok: {tool_call.ok}")
            print(f"   preview: {tool_call.content_preview!r}")
    print()

    print("Usage:")
    print(f"Model: {result.model}")
    print(f"Iterations: {result.iterations}")
    print(f"Input tokens: {result.input_tokens}")
    print(f"Output tokens: {result.output_tokens}")
    print(f"Stop reason: {result.stop_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
