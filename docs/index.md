# LongVideoAgent Docs

LongVideoAgent is a multi-agent framework for reasoning over long videos. A master LLM iteratively coordinates a grounding agent and a vision agent to localize relevant clips, inspect visual evidence, and answer long-video QA questions with interpretable multi-step traces.

[Project Page](https://longvideoagent.github.io/) | [Paper](https://arxiv.org/abs/2512.20618) | [Dataset: LongTVQA](https://huggingface.co/datasets/longvideoagent/LongTVQA) | [Dataset: LongTVQA+](https://huggingface.co/datasets/longvideoagent/LongTVQA_plus)

## Overview

Recent long-video QA systems often rely on lossy summarization or limited tool use, which weakens temporal grounding and misses fine-grained cues. LongVideoAgent instead uses a bounded multi-agent reasoning loop:

- The **MasterAgent** plans the next step and decides when enough evidence has been collected.
- The **GroundingAgent** localizes question-relevant video clips from subtitles.
- The **VisionAgent** inspects sampled frames from the localized clips and returns targeted visual observations.

This repository documents the codebase for:

- Running **quickstart GRPO training** for the master agent.
- Building **offline grounding caches** and converting datasets into training-ready formats.
- Running **unified local/API evaluation** on LongTVQA and LongTVQA+.

## Documentation Guide

- Start with [Installation](installation.md) to set up the environment.
- Use [Quickstart](training/quickstart.md) for the shortest end-to-end training path.
- See [Evaluation](evaluation.md) for local and API-based evaluation scripts.
- Use the training pages for GRPO config details, offline grounding cache generation, and LoRA adapter merging.

## Method Summary

LongVideoAgent operates in a bounded loop with up to `K` reasoning rounds. At each step, the master agent emits a structured action:

- `<request_grounding>` to search for relevant clips.
- `<visual_query>` to inspect selected clips with the vision agent.
- `<answer>` to terminate and return the final option.

The master agent is optimized with reinforcement learning so that trajectories remain structurally valid, concise, and correct.
