# 🎬LongVideoAgent: Multi-Agent Reasoning with Long Videos

<div align="center">

[中文](./README_zh.md) | [Project Page](https://longvideoagent.github.io/) | [Arxiv](https://arxiv.org/abs/2512.20618)

Runtao Liu\*, Ziyi Liu\*, Jiaqi Tang, Yue Ma, Renjie Pi, Jipeng Zhang, Qifeng Chen

Hong Kong University of Science and Technology

\* Equal contribution

</div>

---

This is the official repository for [arXiv](https://arxiv.org/abs/2512.20618). Code and dataset are coming soon.

## 🚀 Latest News
• `[2025/12/24]:` 🚀 We released our paper "LongVideoAgent: Multi-Agent Reasoning with Long Videos" on [arXiv](https://arxiv.org/abs/2512.20618)!

## 📅 Roadmap
- [ ] **[2026/01/01]**: Release **LongTVQA** & **LongTVQA+** datasets.
- [ ] **[2026/01 - 2026/02]**: Release code for inference and training.

---

## 📝 Abstract
Recent advances in multimodal LLMs and systems that use tools for long-video QA point to the promise of reasoning over hour-long episodes. However, many methods still compress content into lossy summaries or rely on limited toolsets, weakening temporal grounding and missing fine-grained cues. We propose **LongVideoAgent**, a multi-agent framework in which a master LLM coordinates a grounding agent to localize question-relevant segments and a vision agent to extract targeted textual observations. The master agent plans with a step limit, and is trained with reinforcement learning to encourage concise, correct, and efficient multi-agent cooperation. This design helps the master agent focus on relevant clips via grounding, complements subtitles with visual detail, and yields interpretable trajectories. On our proposed **LongTVQA** and **LongTVQA+** which are episode-level datasets aggregated from TVQA/TVQA+, our multi-agent system significantly outperforms strong non-agent baselines. Experiments also show reinforcement learning further strengthens reasoning and planning for the trained agent.

---

## 🤖 Method: Multi-Agent Framework
![Architecture](https://arxiv.org/html/2512.20618v1/x2.png)

Architecture of **LongVideoAgent**. A **MasterAgent** runs for up to $K$ rounds, collaborating with a **GroundingAgent** to localize relevant clips from videos and a **VisionAgent** to read fine-grained cues from the localized frames. Evidence accumulates until the **MasterAgent** feels confident to answer the user.

### 🔄 Iterative Reasoning Loop
Unlike single-pass models, LongVideoAgent operates in a bounded loop (max $K$ steps). At each step, the **MasterAgent** generates a "thinking" trace and emits a structured action token:
- `<request_grounding>`: Calls the **GroundingAgent** to localize relevant video segments based on subtitles. The agent returns a symbolic tag `<clip_X>`.
- `<visual_query>`: Calls the **VisionAgent** to extract specific visual details (objects, actions, text) from the localized clip. The agent returns textual observations.
- `<answer>`: Terminates the loop and provides the final response when sufficient evidence is gathered.

### 🧠 Reinforcement Learning (GRPO)
We optimize the MasterAgent using Group Relative Policy Optimization (GRPO). The training objective includes: 1.  Structural validity. 2.  Answer Correctness: Rewarding the agent for reaching the correct final answer.

---

## 📈 Experimental Results
We evaluate LongVideoAgent on **LongTVQA** and **LongTVQA+**, which are episode-level datasets.

### Main Results
Performance on *LongTVQA* and *LongTVQA+*. The left block lists model attributes (*Agentic*, *Input*, *RL fine-tune*); the right block reports validation accuracy (%). GPT-4o and Gemini-2.5 Pro are *multimodal* baselines that process and accept the full long video directly. Methods labeled `Agentic` indicate the model operates as the MasterAgent; methods labeled `AgenticRL` additionally denote RL fine-tuning. Parenthesized green numbers denote absolute gains over the immediately preceding (non-agentic or non-RL) setting. We observe that: (i) our multi-agent framework, **LongVideoAgent**, consistently outperforms the non-agentic counterparts; (ii) agentic RL yields additional gains, especially for smaller open-source models; (iii) using frames provides visual evidence beyond subtitles, and generally outperforms subtitle-only inputs; (iv) closed-source models remain strong, but the gap narrows much when open-source models adopt agentic designs and agentic RL.

<p align="center">
  <img src="readme_src/main_result.png" width="100%" />
</p>

### 🔍 Ablation Analysis
We conduct comprehensive ablation studies to validate our design choices. First, we observe that both grounding and vision agents are essential, with the full multi-agent system achieving the highest accuracy. Second, increasing the reasoning step limit $K$ improves performance until saturation, confirming the value of iterative planning. Finally, stronger vision backbones and larger temporal windows provide richer context, further boosting the agent's reasoning capabilities.

<p align="center">
  <img src="readme_src/ablation.png" width="50%" />
</p>

---

## 📝 Citation
If you find our work helpful, please cite:
```bibtex
@misc{liu2025longvideoagentmultiagentreasoninglong,
      title={LongVideoAgent: Multi-Agent Reasoning with Long Videos}, 
      author={Runtao Liu and Ziyi Liu and Jiaqi Tang and Yue Ma and Renjie Pi and Jipeng Zhang and Qifeng Chen},
      year={2025},
      eprint={2512.20618},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.20618}, 
}
```

---



