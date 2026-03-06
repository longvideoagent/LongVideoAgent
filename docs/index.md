# LongVideoAgent

**A multi-agent framework for reasoning over long videos, with a master LLM coordinating grounding and vision agents for efficient, fine-grained video QA.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)  
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)  
[![arXiv](https://img.shields.io/badge/arXiv-2512.20618-b31b1b.svg)](https://arxiv.org/abs/2512.20618)  
[![GitHub stars](https://img.shields.io/github/stars/longvideoagent/LongVideoAgent?style=social)](https://github.com/longvideoagent/LongVideoAgent)

---

## Overview

This repository provides a **complete implementation** of **LongVideoAgent** — a multi-agent system for reasoning over hour-long videos. It addresses the limitations of traditional MLLMs that rely on lossy compression or limited tools, by enabling a **master LLM** to iteratively coordinate:

- A **grounding agent** to localize question-relevant video segments.
- A **vision agent** to extract fine-grained visual observations (objects, actions, faces, etc.).

The framework supports:
- **Training**: Reinforcement learning via **[VERL](https://github.com/volcengine/verl)** to optimize the master agent's planning and multi-agent cooperation.
- **Evaluation**: Full agent pipelines on episode-level benchmarks.
- **Baseline**: API-based evaluation (e.g., Gemini) on TVQA and TVQA+ for rapid prototyping.

Key datasets: **LongTVQA** and **LongTVQA+** (aggregated from TVQA/TVQA+). The system achieves state-of-the-art results with interpretable, multi-round decision traces.
