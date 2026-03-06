#!/usr/bin/env python3
"""Smoke test for OpenAI-compatible grounding API endpoint."""

from __future__ import annotations

import argparse
import os
import sys

from openai import OpenAI


DEFAULT_BASE_URL = "https://api2.aigcbest.top/v1"
DEFAULT_MODEL = "gemini-3-flash-preview"


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for grounding API")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="OpenAI-compatible API base URL")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name to test")
    parser.add_argument("--prompt", type=str, default="你好? 请只回复 OK。", help="User prompt for smoke test")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout in seconds")
    args = parser.parse_args()

    api_key = os.getenv("qdd_api")
    if not api_key:
        print("ERROR: Missing env var qdd_api", file=sys.stderr)
        sys.exit(2)

    client = OpenAI(base_url=args.base_url, api_key=api_key, timeout=args.timeout)

    print(f"[smoke] base_url={args.base_url}")
    print(f"[smoke] model={args.model}")
    print("[smoke] sending request...")

    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": args.prompt}],
        )
    except Exception as exc:
        print(f"[smoke] FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

    content = ""
    if response.choices and response.choices[0].message:
        content = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)

    print("[smoke] SUCCESS")
    print(f"[smoke] response={content.strip()}")
    if usage is not None:
        print(
            "[smoke] usage="
            f"prompt_tokens={getattr(usage, 'prompt_tokens', 'n/a')}, "
            f"completion_tokens={getattr(usage, 'completion_tokens', 'n/a')}, "
            f"total_tokens={getattr(usage, 'total_tokens', 'n/a')}"
        )


if __name__ == "__main__":
    main()
