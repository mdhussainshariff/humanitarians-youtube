# Fellowship Renewal Request — Ameya P.

**Name:** Ameya Padwad  
**Role:** Machine Learning Engineer  
**Project:** Fine tuning using RLCD (`RL Environment` / `AmeyaPadwad/fine_tuner`)  
**Current agreement dates:** 09/14/2026 to 10/31/2026 (September 14, 2026 – October 31, 2026)  
**Requested renewal dates:** 11/01/2026 to 01/31/2027 (November 1, 2026 – January 31, 2027 · 3 months)  
**Fellows folder:** [fellows/ameya-p](https://github.com/nikbearbrown/humanitarians-youtube/tree/main/fellows/ameya-p)

---

## Completed

1. **End-to-End Reinforcement Learning with Calibrated Decisions (RLCD) Runtime & 60,000-Row xLAM Validation on Laya**  
   Designed and implemented a complete, modular RLCD fine-tuning runtime (`helper.py`, `models.py`, `reward_strategies/`, `rl_training.py`) for the non-autoregressive `convaiinnovations/laya` decision encoder, combining Group Relative Policy Optimization (GRPO) via Gaussian logit perturbations ($G=4$), strictly proper Brier scoring rewards ($R = -(c - y)^2$), and post-training L-BFGS temperature scaling (`temperature.pt`). Validated at scale on the 60,000-row `Salesforce/xlam-function-calling-60k` benchmark, boosting held-out tool-selection routing accuracy from a **~73% baseline to 99.10%** while driving Expected Calibration Error (ECE) down to **0.83%**.
   - **My contribution:** Sole author of the RLCD mathematical formulation, dataset/marker-packing adapters, custom `Trust_remote_code` model loader, GRPO training engine, 4-stage CPU verification suite (`smoke_tests.py`), and full-scale GPU validation runs.
   - **Evidence:** [Repository (`AmeyaPadwad/fine_tuner`)](https://github.com/AmeyaPadwad/fine_tuner) · [PR #1 / Commit `b3c6513`](https://github.com/AmeyaPadwad/fine_tuner/commit/b3c6513) · [Week 1 Report](../2026-09-21-project-overview/work_report.md) · [Week 2 Report](../2026-09-28-laya-rlcd-implementation/work_report.md) · [Week 3 Validation Report](../2026-10-05-xlam-laya-validation/work_report.md) · [Week 3 Video (16:9)](https://drive.google.com/file/d/18QoVeYQo5DZb1EzCfQrX_jtsI49dC3Zh/view?usp=drive_link)

2. **Dataset-Agnostic & Model-Agnostic Architecture + Automated Post-Hoc Calibration Bake-Off Across 5 Decision Models**  
   Engineered automatic schema inference (`rlcd/processing.py`: `auto_prepare_dataset()` mapping arbitrary Hugging Face datasets into `choice`, `noul`, and `score` primitives) and unified model wrapping (`rlcd/models.py`: `MarkerDecisionModelWrapper` across `Laya`, `Von`, `OpenDecision`, `Typical-small`, and `Decision 1.0`). Executed an 8-run multi-GPU (2× NVIDIA T4 DDP) benchmark suite across 4 datasets (`lmsys/toxic-chat`, `mteb/tweet_sentiment`, `Anthropic/hh-rlhf`, `NexusRaven_API_eval`) where Mean Brier Score improved in **8/8 runs** and ECE dropped in **7/8 runs**. To resolve post-training multi-class ECE spikes on subjective tasks, built an automated 4-strategy **Post-Hoc Calibration Bake-Off** (`Raw Logits`, `Temperature Scaling`, `Platt / Vector Scaling`, and `Matrix Scaling` via smooth log-space L-BFGS) that slashed ECE by an average of **73%**—bringing `wfzyx/von` Tweet Sentiment (`17.24% → 4.85%`), `wfzyx/von` Anthropic RLHF (`18.46% → 4.92%`), and `Decision 1.0` Tweet Sentiment (`17.38% → 4.76%`) all below 5% in under 2 seconds on CPU while preserving **100% of fine-tuned accuracy**.
   - **My contribution:** Sole engineer of `rlcd/processing.py`, `rlcd/models.py`, and the post-hoc calibration tournament (`training.py`, `run_post_hoc_calibration.py`); executed all multi-GPU benchmark runs and diagnosed causal-head vs. multi-class calibration dynamics.
   - **Evidence:** [PR #2 / Commit `acf4851`](https://github.com/AmeyaPadwad/fine_tuner/commit/acf4851) · [Benchmark Results Commit `2638881`](https://github.com/AmeyaPadwad/fine_tuner/commit/2638881) · [Post-Hoc Calibration Commit `7babb8c`](https://github.com/AmeyaPadwad/fine_tuner/commit/7babb8c) · [Smooth L-BFGS Fix `a625c6f`](https://github.com/AmeyaPadwad/fine_tuner/commit/a625c6f) · [Week 4 Report](../2026-10-12-agnostic-calibration-benchmarks/work_report.md) · [Week 5 Report](../2026-10-19-post-hoc-calibration-agent-sdk/work_report.md)

3. **Installable `rlcd` Python SDK, CLI (`rlcd-fine-tune`), Autonomous Agent Skill (`SKILL.md`), and 5 Brutalist 4K Video Reels**  
   Packaged the platform into an installable Python SDK (`rlcd.fine_tune()` with 32 configurable training, model, schema, and calibration parameters), CLI executable (`rlcd-fine-tune`), and plug-and-play autonomous agent skill (`skills/rlcd-fine-tune/SKILL.md`), validated end-to-end by an isolated AI agent across a 3-phase GPU test suite (`tests/test_agent.md`, `results/SDK_agent_testing_report.md`) with zero human code edits. Produced 5 weekly Brutalist 4K (16:9) explainer and progress reels, each with a verified `beat_sheet.json`, `FACTCHECK.md`, `TYPECHECK.md` (`GATE T` 0 FAILs), `FRICTIONAL.md`, and `HOURS.md`.
   - **My contribution:** Packaged `pyproject.toml` and `__init__.py`, authored `skills/rlcd-fine-tune/SKILL.md` and the 3-phase agent evaluation harness, and scripted, rendered, and verified all 5 weekly Brutalist reels.
   - **Evidence:** [SDK & Skill Commit `6ce866d`](https://github.com/AmeyaPadwad/fine_tuner/commit/6ce866d) · [Agent Verification Report Commit `07d36c7`](https://github.com/AmeyaPadwad/fine_tuner/commit/07d36c7) · [Video Drive Folder](https://drive.google.com/drive/folders/1gH4FUFMF_hLI9aItdn7q8SeatKn2BRNc) · [Fellows Folder Index](../README.md)

---

## Hours Status

**20+ hours documented for every reported week** across Weeks 1–5 (**114.5 hours total**: Week 1 = 21.0 hrs, Week 2 = 24.0 hrs, Week 3 = 21.5 hrs, Week 4 = 23.5 hrs, Week 5 = 24.5 hrs). Every week meets or exceeds the 20-hour minimum and includes documented engineering/research work, Brutalist video production, and weekly team/supervisor meetings (see [HOURS.md](../HOURS.md) and each week's per-folder `HOURS.md`). No missing or incomplete records for Weeks 1–5; the final week of the current agreement (October 26–31, 2026) is in progress at the time of writing.

---

## Weekly Evidence Index

| Week / dates                       | Actual hours | Video links                                                                                                                                                                                                                       | Log and work evidence                                                                                                                                                                                                                                                                                                                                                                                         |
| ---------------------------------- | -----------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Week 1** · Sep 21 – Sep 25, 2026 |         21.0 | [Project Overview Video (16:9)](https://drive.google.com/file/d/1gnYaMmkuXmg6rN2-qe5dR_NawbfcE5Jh/view?usp=drive_link) · [Drive Folder](https://drive.google.com/drive/folders/1gH4FUFMF_hLI9aItdn7q8SeatKn2BRNc)                 | [Frictional log](../2026-09-21-project-overview/FRICTIONAL.md) · [Work report](../2026-09-21-project-overview/work_report.md) · [Weekly hours](../2026-09-21-project-overview/HOURS.md) · [Folder](../2026-09-21-project-overview/)                                                                                                                                                                           |
| **Week 2** · Sep 28 – Oct 4, 2026  |         24.0 | [Laya & RLCD Implementation Video (16:9)](https://drive.google.com/file/d/12uyzcsO7FPG8bhyxweBCAoCdTA_a8whp/view?usp=drive_link) · [Drive Folder](https://drive.google.com/drive/folders/1gH4FUFMF_hLI9aItdn7q8SeatKn2BRNc)       | [Frictional log](../2026-09-28-laya-rlcd-implementation/FRICTIONAL.md) · [Work report](../2026-09-28-laya-rlcd-implementation/work_report.md) · [Weekly hours](../2026-09-28-laya-rlcd-implementation/HOURS.md) · [PR #1 (`b3c6513`)](https://github.com/AmeyaPadwad/fine_tuner/commit/b3c6513)                                                                                                               |
| **Week 3** · Oct 5 – Oct 11, 2026  |         21.5 | [xLAM & Laya Validation Video (16:9)](https://drive.google.com/file/d/18QoVeYQo5DZb1EzCfQrX_jtsI49dC3Zh/view?usp=drive_link) · [Drive Folder](https://drive.google.com/drive/folders/1gH4FUFMF_hLI9aItdn7q8SeatKn2BRNc)           | [Frictional log](../2026-10-05-xlam-laya-validation/FRICTIONAL.md) · [Work report](../2026-10-05-xlam-laya-validation/work_report.md) · [Weekly hours](../2026-10-05-xlam-laya-validation/HOURS.md) · [Folder](../2026-10-05-xlam-laya-validation/)                                                                                                                                                           |
| **Week 4** · Oct 12 – Oct 18, 2026 |         23.5 | [Agnostic Calibration Benchmarks Video (16:9)](https://drive.google.com/file/d/1vSt0JxMK_NULaiTBgMjvm54r55B4b0Qm/view?usp=drive_link) · [Drive Folder](https://drive.google.com/drive/folders/1gH4FUFMF_hLI9aItdn7q8SeatKn2BRNc)  | [Frictional log](../2026-10-12-agnostic-calibration-benchmarks/FRICTIONAL.md) · [Work report](../2026-10-12-agnostic-calibration-benchmarks/work_report.md) · [Weekly hours](../2026-10-12-agnostic-calibration-benchmarks/HOURS.md) · [PR #2 (`acf4851`)](https://github.com/AmeyaPadwad/fine_tuner/commit/acf4851) · [Benchmarks (`2638881`)](https://github.com/AmeyaPadwad/fine_tuner/commit/2638881)     |
| **Week 5** · Oct 19 – Oct 25, 2026 |         24.5 | [Post-Hoc Calibration & Agent SDK Video (16:9)](https://drive.google.com/file/d/1MlWelhpzkbFDfiFVIqieMH__K0b_p_Yr/view?usp=drive_link) · [Drive Folder](https://drive.google.com/drive/folders/1gH4FUFMF_hLI9aItdn7q8SeatKn2BRNc) | [Frictional log](../2026-10-19-post-hoc-calibration-agent-sdk/FRICTIONAL.md) · [Work report](../2026-10-19-post-hoc-calibration-agent-sdk/work_report.md) · [Weekly hours](../2026-10-19-post-hoc-calibration-agent-sdk/HOURS.md) · [Calibration (`7babb8c`)](https://github.com/AmeyaPadwad/fine_tuner/commit/7babb8c) · [SDK & Skill (`6ce866d`)](https://github.com/AmeyaPadwad/fine_tuner/commit/6ce866d) |
| **Total (Weeks 1–5)**              |    **114.5** |                                                                                                                                                                                                                                   |                                                                                                                                                                                                                                                                                                                                                                                                               |

---

## Degree Relevance and Supervision

This fellowship directly applies and extends my **Master of Science in Electrical and Computer Engineering** from **Northeastern University** (Boston, MA; graduated December 2025, GPA: 3.76), with a concentration in **Computer Vision, Machine Learning and Algorithms** and graduate coursework in _Reinforcement Learning and Decision Making Under Uncertainty_, _Machine Learning and Pattern Recognition_, _Verifiable Machine Learning_, _Computer Vision_, and _Data Visualization_, as well as graduate research on _Improving Formal Verification Gaps in Neural Network-Based Controllers_. Designing GRPO policy-gradient algorithms with strictly proper scoring rules, optimizing multi-class probabilistic calibration (Brier score, Expected Calibration Error, temperature/Platt/matrix scaling), and verifying autonomous agent execution across neural decision architectures directly exercise my graduate training in reinforcement learning under uncertainty and verifiable machine learning.

**Supervisors:** Dev Shah and Dhruv. Weekly progress, architectural decisions, and benchmark results were reviewed across weekly sync meetings and pull request reviews in [`AmeyaPadwad/fine_tuner`](https://github.com/AmeyaPadwad/fine_tuner).

---

## Authorization

- **Authorization type:** F-1 Post-Completion OPT

---

## Next Period Research & Engineering Plan

**Requested renewal period:** November 1, 2026 – January 31, 2027 (3 months).

**Research focus (Research Frontiers — Open Projects):**  
The current fine-tuning platform project is now complete and concluded. The next three-month period (`November 1, 2026 – January 31, 2027`) will focus on taking up a new, open-ended research question under one of the **Research Frontiers — Open Projects** verticals:

1. **Intelligence Measurement** — Finding capabilities that cannot yet be reliably measured, inventing evaluation methods and benchmarks, testing competing systems, and exposing fundamental limitations.
2. **Learning Environments** — Identifying behaviors worth teaching and building environments, rewards, simulations, and challenges that produce better learning.
3. **Learning Algorithms** — Investigating where existing training methods fail and inventing new optimization techniques, learning rules, or adaptation mechanisms.
4. **Cognitive Architectures** — Exploring how intelligence should be structured across reasoning, memory, planning, and decision-making architectures.
5. **Security & Exploitation** — Investigating weaknesses in real systems, reproducing vulnerabilities in authorized targets, and developing defenses.
6. **Data & Knowledge** — Developing methods to discover, generate, organize, verify, and connect missing, inaccessible, or unreliable information.
7. **Compute & Execution** — Identifying fundamental bottlenecks in computation across compilers, inference, distributed systems, and hardware utilization.
8. **Formal Proof & Verification** — Formalizing mathematical, software, or engineering claims that are difficult to guarantee, constructing proofs or counterexamples, and developing automatic verification methods.

Following the Research Frontiers 5-stage approach (_1. Explore_ a vertical, study existing limitations, identify 3–5 unsolved problems, and obtain lead approval to avoid conflicts or repetition; _2. Choose_ an important, overlooked, or misunderstood problem and form a testable hypothesis; _3. Experiment_ by building, testing ideas, challenging assumptions, and following unexpected findings; _4. Validate_ with reproducible evidence against existing approaches while documenting informative negative results; and _5. Deliver_ reusable knowledge and artifacts), the next period will produce a combination of **`SKILL.md`** specifications, **working artifacts** (code, benchmarks, environments, algorithms, datasets, or prototypes), **research findings**, **learned processes**, and **research reports**.

| Target date                                     | Deliverable                                                                                                                                                                                                                                                                              |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **November 15, 2026** (_Explore & Choose_)      | Select a target Research Vertical, study existing work and limitations, identify 3–5 candidate unsolved problems, obtain lead approval on the chosen research question to avoid conflicts/repetition, and formulate a testable hypothesis.                                               |
| **December 31, 2026** (_Experiment & Validate_) | Build the experimental prototype, benchmark, environment, or algorithm for the selected research question; run iterative experiments against existing baselines; and document empirical findings (including unexpected behavior and negative results).                                   |
| **January 31, 2027** (_Deliver_)                | Complete reproducible validation and deliver the full Research Frontiers output package—reusable `SKILL.md`, working code/artifacts, learned research methodology, and comprehensive research report—alongside weekly Brutalist 4K videos, `FRICTIONAL.md` logs, and `HOURS.md` records. |

---

---
