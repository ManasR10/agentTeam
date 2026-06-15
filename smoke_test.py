from __future__ import annotations

import sys

from llm import call_llm

EXPECTED_REPLY = "NPW Phase 0 setup OK"


def main() -> int:
    print("Running NPW Phase 0 smoke test...")
    try:
        result = call_llm(
            prompt=(
                "Reply with exactly the following text and nothing else:\n"
                f"{EXPECTED_REPLY}"
            ),
            system=(
                "You are a deterministic API smoke-test assistant. "
                "Follow the user's requested output format exactly."
            ),
            max_tokens=32,
        )
    except Exception as exc:
        print("\nSmoke test failed.")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error: {exc}")
        return 1

    if result.text != EXPECTED_REPLY:
        print("\nSmoke test failed: unexpected response.")
        print(f"Expected: {EXPECTED_REPLY!r}")
        print(f"Received: {result.text!r}")
        return 1

    print()
    print(result.text)
    print()
    print(f"Model: {result.model}")
    print(f"Input tokens: {result.input_tokens}")
    print(f"Output tokens: {result.output_tokens}")
    print(f"Stop reason: {result.stop_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
