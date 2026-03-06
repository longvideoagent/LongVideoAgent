#!/usr/bin/env python3
"""Minimal local vLLM inference smoke test."""

from __future__ import annotations

import argparse
import sys
from typing import Optional


def build_prompt(question: str) -> str:
    return (
        "<|im_start|>system\nYou are a concise assistant.<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def run_test(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    tensor_parallel_size: int,
    dtype: str,
    trust_remote_code: bool,
) -> str:
    try:
        from vllm import LLM, SamplingParams
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import vllm. Install it first, e.g. `pip install vllm`."
        ) from e

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    outputs = llm.generate([prompt], sampling_params)
    if not outputs or not outputs[0].outputs:
        raise RuntimeError("vLLM returned empty outputs.")
    return outputs[0].outputs[0].text.strip()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="vLLM local inference smoke test")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Instruct-3B", help="HF model id")
    parser.add_argument("--question", default="用一句话解释什么是强化学习。", help="User question")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto", help="auto, float16, bfloat16, float32")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)

    prompt = build_prompt(args.question)
    print(f"[vllm-smoke-test] model={args.model}")
    try:
        answer = run_test(
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception as e:
        print(f"[vllm-smoke-test] FAILED: {e}", file=sys.stderr)
        return 1

    print("[vllm-smoke-test] SUCCESS")
    print("=== prompt ===")
    print(args.question)
    print("=== output ===")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

