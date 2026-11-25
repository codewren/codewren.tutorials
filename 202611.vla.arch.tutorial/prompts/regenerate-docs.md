# Prompt: regenerate the VLA architecture tutorial

Copy everything below the line into a new agent session started in this project folder.

---

Write a chapter-based tutorial under `docs/` that explains the architecture of Vision-Language-Action (VLA) models, with a focus on **inference**. Cover all essential components and the popular ones. Write Chapter 0 first (overview and outline), then write every other chapter based on Chapter 0.

## Files

- File name pattern: `ch-N-title.md` (use `.pdf` / `.ppt` only if the file type requires it).
- Chapters:
  - `ch-0-overview.md`
  - `ch-1-inference-loop.md`
  - `ch-2-inputs-and-encoders.md`
  - `ch-3-vlm-backbone.md`
  - `ch-4-action-heads.md`
  - `ch-5-action-execution.md`
  - `ch-6-dual-system-architectures.md`
  - `ch-7-inference-optimization.md`
  - `ch-8-end-to-end-walkthrough.md`

## Chapter content

- **Ch 0 — Overview:** what a VLA is, the six-stage pipeline (inputs → encoders → backbone → action head → execution → control), a linked chapter table, an essential-vs-popular components table, the **Notation** table, the running example, and a reference-model table (RT-2, Octo, OpenVLA, π0, π0-FAST, OpenVLA-OFT, GR00T N1, Helix, π0.5, SmolVLA).
- **Ch 1 — The Inference Loop:** the closed perception-action loop; the three rates (inference rate, action rate, controller rate); the latency budget (π0's reported 14 + 32 + 27 = 73 ms on-board, 86 ms off-board); idle time when the robot waits for the model.
- **Ch 2 — Inputs and Encoders:** ViT patching; SigLIP, DINOv2 and fused encoders; tokenizer and prompt template; proprioception token; image-token reduction.
- **Ch 3 — The VLM Backbone:** projector; LLM decoder; fully causal vs. block-wise causal masks; π0's action expert (two weight sets sharing one attention); KV cache, prefill vs. decode.
- **Ch 4 — Action Heads:** four families (discrete bins, FAST = DCT + BPE, parallel regression as in OpenVLA-OFT, diffusion / flow matching); action normalization; action chunking.
- **Ch 5 — Action Execution:** de-normalization; conversion to controller format; receding horizon; temporal ensembling; synchronous vs. asynchronous execution; Real-Time Chunking; safety layer and low-level controller.
- **Ch 6 — Dual-System Architectures:** System 1 / System 2; interface options (latent, features, language subtask); Helix; GR00T N1; π0.5 / Hi Robot.
- **Ch 7 — Inference Optimization:** compute-bound prefill (`F ≈ 2·N_θ·N_pre`); memory-bound decode (`≥ N_θ·β/BW`); a technique table; deployment topologies (on-board, off-board, hybrid).
- **Ch 8 — End-to-End Walkthrough:** one inference step traced with shapes for the π0-style running example and for an OpenVLA-style model; a model comparison table; an 8-question design checklist.

## Notation and running example

- Chapter 0 defines a Notation table. All chapters use **only** those symbols, each with **one** meaning. A tilde (`ã`) marks a normalized quantity.
- Use one running example (π0-style) in every chapter:
  - 1 arm, `D = 7`, `D_q = 8`
  - `C = 3` cameras, `R = 224`, `P = 14` → `N_p = 256`, `N_img = 768`
  - `N_txt = 8` → `N_pre = 776`
  - `H = 50`, `N_suf = 51`, `S = 10`, `δ = 0.1`, `B = 256`
  - `d_v = 1152`, `d = 2048`, `d_e = 1024`, `L = 18`, `n_h = 8`, `n_kv = 1`, `d_h = 256`, `β = 2`, `BW = 10¹²` B/s
  - `f_a = 50 Hz`, `n = 25`, `T_inf = 0.073 s`
  - Helix-like rates `f_S1 = 200 Hz`, `f_S2 = 8 Hz`
  - Comparison model: OpenVLA, `N_θ ≈ 7.5×10⁹`

## Chapter layout (every chapter)

1. H1 "Chapter N — Title"
2. Short intro
3. Diagram(s)
4. Formulas
5. Running-example numbers
6. A tiny hand-checkable worked example
7. A short NumPy snippet (optional)
8. "Key takeaways" bullets
9. "References"
10. Footer with ← previous / next → chapter links. The last chapter's "next" link goes back to Chapter 0.

## Diagrams

- Use Mermaid, always horizontal: `flowchart LR`, and `direction LR` inside every subgraph.
- ASCII diagrams are allowed for matrices and shapes.

## References

- Each chapter lists the published papers, textbooks or blogs it took information from. Keep at most the **10 most important** per chapter.
- Papers must be published in a top conference in the field (e.g. NeurIPS, ICML, ICLR, CoRL, RSS, ICRA, ICCV, ACL, EMNLP, MLSys, SOSP) or in a high-reputation journal.
- Blogs must come from a well-known company or organization in the field (e.g. Figure AI for Helix, NVIDIA for GR00T N1, Hugging Face for SmolVLA, Google for PaliGemma).
- Do not cite arXiv-only papers. If a fact has only an arXiv source, say so in a one-line note under the reference list.
- Look up publication venues online. Do not rely on memory.

## Verification

- Verify **all** arithmetic by running it before writing it down.
- Parse every Mermaid block. Check that it is `LR` and that every subgraph has `direction LR`.
- Check that every chapter link resolves.
- Report anything you could not verify, such as NumPy snippets you could not execute.
