# Summary and regeneration prompt

## Summary of the docs

A 13-chapter Markdown tutorial in `docs/` that follows one prompt through a decoder-only
Transformer LLM during **inference**, from raw text in to the next token out. All shapes and
numbers use a Llama-2-7B-like running example.

| Ch | File | What it explains |
|----|------|------------------|
| 0 | `ch-0-overview.md` | Pipeline and block diagrams, chapter map with links, Notation table (`seq`, `ctx`, `d_model`, …), how prefill and decode take turns |
| 1 | `ch-1-tokenizer.md` | Sub-word tokenization, BPE training/encoding with a worked example, vocabularies, special tokens |
| 2 | `ch-2-token-embedding.md` | Embedding table lookup → `[seq, d_model]` residual stream; size; weight tying |
| 3 | `ch-3-positional-encoding.md` | Why position is needed; sinusoidal/learned absolute encodings; RoPE and its relative-position property |
| 4 | `ch-4-normalization.md` | LayerNorm vs. RMSNorm with a worked example; pre-norm vs. post-norm |
| 5 | `ch-5-self-attention.md` | Q/K/V, scaled dot-product attention, causal mask, prefill vs. decode shapes, cost |
| 6 | `ch-6-multi-head-attention.md` | Splitting into heads, `W_O`, shapes; MHA vs. GQA vs. MQA and the KV cache |
| 7 | `ch-7-feed-forward-network.md` | Classic FFN and SwiGLU; `[seq, d_model]` input processed row by row; cost; 1- and 2-token worked examples |
| 8 | `ch-8-decoder-block.md` | Assembling a block; residual stream; stacking `n_layers` blocks; parameter count |
| 9 | `ch-9-mixture-of-experts.md` | Router + top-k experts per token; total vs. active parameters; inference concerns |
| 10 | `ch-10-lm-head-and-sampling.md` | LM head on the last row; softmax; greedy, temperature, top-k, top-p with worked numbers |
| 11 | `ch-11-inference-loop.md` | Prefill vs. decode; KV cache size; why decode is memory-bound; speed-up techniques |
| 12 | `ch-12-putting-it-together.md` | 20-step shape trace of a 6-token prompt; full parameter, FLOP and memory budget (6.74B params) |

**Conventions used throughout:** left-to-right Mermaid diagrams; one meaning per symbol (defined
in Ch 0); every chapter has a worked example, a short NumPy snippet, key takeaways, a
`## References` list (at most 10 peer-reviewed papers or textbooks) and previous/next links.

**Known gaps (not verified in the session that produced the docs):**
- The NumPy snippets were not executed (NumPy was not installed); the arithmetic and the
  pure-Python BPE snippet were checked.
- Mermaid diagrams were not render-tested.
- References were written from memory and not checked online.
- Facts that come only from technical reports are stated but not cited: the Llama-2-7B
  configuration, GPT-2 / Llama-2 / Llama-3 vocabulary sizes, Mixtral and DeepSeek-V3 parameter
  counts, Mistral/Falcon attention variants, and the original LayerNorm, GELU and SwiGLU papers.

## Prompt to regenerate the docs

Copy the block below into a coding agent, run from the project root.

```text
Write a tutorial, as Markdown files under `docs/`, that explains how a decoder-only
Transformer LLM works during INFERENCE: from raw text in to the next token out.
Audience: engineers new to LLM internals. Tone: concise, tutorial-style.

FILES (exact names, 13 files)
  ch-0-overview.md                ch-1-tokenizer.md               ch-2-token-embedding.md
  ch-3-positional-encoding.md     ch-4-normalization.md           ch-5-self-attention.md
  ch-6-multi-head-attention.md    ch-7-feed-forward-network.md    ch-8-decoder-block.md
  ch-9-mixture-of-experts.md      ch-10-lm-head-and-sampling.md   ch-11-inference-loop.md
  ch-12-putting-it-together.md
  (pattern: ch-N-title.ext, N without zero-padding, title in lowercase-hyphenated form)

GLOBAL RULES
- Diagrams: Mermaid, ALWAYS horizontal (`flowchart LR`, and `direction LR` inside
  subgraphs). ASCII diagrams are allowed for matrices/shapes.
- Running example: Llama-2-7B-like config. vocab=32,000, d_model=4,096, n_layers=32,
  n_heads=32, n_kv_heads=32, d_h=128, d_ff=11,008 (SwiGLU), max_seq=4,096, RMSNorm, RoPE,
  no biases, fp16 weights.
- Use ONLY the symbols defined in the Chapter 0 Notation table, with one meaning each:
  seq, ctx, max_seq, vocab, d_model, n_layers, n_heads, n_kv_heads, d_h, d_ff,
  n_experts, k, batch. Never reuse a letter for two things (e.g. no `N` for both layers
  and experts, no `h` for both heads and hidden state, no `g` for KV heads). If shorthand
  `d` is used, say `d = d_model` right there.
- Tensor names: X : [seq, d_model] = all tokens in the pass (one row per token);
  x : [d_model] = one row of X; h : [seq, d_model] = residual stream after the attention
  sub-layer; Y : [seq, d_model] = output of a block/sub-layer, y = one row of it.
  Shapes are [rows, columns].
- Whenever a layer is described per token (x : [d_model]), also state that the real input
  is X : [seq, d_model] and explain that each row is processed independently with the same
  weights (applies to norms, FFN, MoE, LM head). Attention is the only layer that mixes rows.
- seq vs ctx: seq = tokens in the current forward pass (prefill: prompt length; decode: 1).
  ctx = tokens processed so far = KV-cache length (prefill: ctx = seq). Attention scores are
  [seq, ctx]: [seq, seq] in prefill, [1, ctx] in decode. Decode attention cost uses ctx.
- Every chapter 2–12 starts, right under the H1, with this line:
  > Symbols such as `seq`, `ctx` and `d_model` are defined in [Chapter 0 — Notation](ch-0-overview.md#notation).
- Chapter layout: H1 "Chapter N — Title", short intro, diagram(s), formulas with tensor
  shapes, running-example numbers, a tiny hand-checkable worked example, a short code
  snippet, "Key takeaways" bullets, "## References", then a footer with ← previous /
  next → chapter links (Ch 12 links forward back to Ch 0; Ch 0 has no footer).
- Verify ALL arithmetic (run it) before writing it down.
- REFERENCES: every chapter (including Ch 0) ends, just before the footer, with
  "## References", intro line "The most important sources for this chapter, in order of
  importance.", then at most 10 items: the original source of the chapter's main idea first,
  then sources of specific claims, then textbooks/further reading. Papers MUST be published
  in a top conference (NeurIPS, ICML, ICLR, ACL, EMNLP, NAACL, EACL, AAAI, CVPR, ECCV,
  ICASSP, MLSys, OSDI, SOSP) or a high-reputation journal (JMLR, TACL, Neural Computation,
  Neural Networks, Neurocomputing, Cognitive Science, CACM). Textbooks are allowed. NO
  arXiv-only preprints, technical reports (incl. GPT-2, LLaMA, Llama 2/3, Mistral, Mixtral,
  DeepSeek-V3, Falcon), blog posts or workshop papers; facts from those may stay in the
  text but are not cited. Format: "Authors (Year). *Title*. Venue. [arXiv:ID](https://arxiv.org/abs/ID)
  — what it supports." Never invent a reference or an arXiv ID. Use these lists:
  Ch 0: Vaswani 2017 Attention Is All You Need (NeurIPS); Brown 2020 GPT-3 (NeurIPS);
        Chowdhery 2023 PaLM (JMLR); Bengio 2003 neural probabilistic LM (JMLR); Pope 2023
        Efficiently Scaling Transformer Inference (MLSys); Hoffmann 2022 compute-optimal LLMs
        (NeurIPS); Jurafsky & Martin SLP 3rd ed.; Raschka 2024 Build a LLM (From Scratch);
        Goodfellow 2016 Deep Learning; Zhang 2023 Dive into Deep Learning.
  Ch 1: Sennrich 2016 BPE (ACL); Kudo & Richardson 2018 SentencePiece (EMNLP demos); Kudo 2018
        subword regularization (ACL); Wang 2020 byte-level subwords (AAAI); Schuster &
        Nakajima 2012 WordPiece (ICASSP); Devlin 2019 BERT (NAACL); Provilkov 2020 BPE-Dropout
        (ACL); Rust 2021 How Good is Your Tokenizer (ACL); Jurafsky & Martin; Raschka 2024.
  Ch 2: Bengio 2003 (JMLR); Vaswani 2017; Mikolov 2013 distributed word representations
        (NeurIPS); Pennington 2014 GloVe (EMNLP); Press & Wolf 2017 weight tying (EACL);
        Inan 2017 tying (ICLR); Jurafsky & Martin; Goodfellow 2016; Zhang 2023 D2L.
  Ch 3: Su 2024 RoFormer (Neurocomputing); Vaswani 2017; Gehring 2017 conv seq2seq (ICML);
        Chowdhery 2023 PaLM; Shaw 2018 relative positions (NAACL); Ke 2021 TUPE (ICLR);
        Press 2022 ALiBi (ICLR); Peng 2024 YaRN (ICLR); Kazemnejad 2023 length generalization
        (NeurIPS); Zhang 2023 D2L.
  Ch 4: Zhang & Sennrich 2019 RMSNorm (NeurIPS); Xiong 2020 LayerNorm placement (ICML);
        Liu 2020 difficulty of training Transformers (EMNLP); Vaswani 2017; Wang 2019 deep
        Transformers (ACL); Brown 2020 GPT-3; Dehghani 2023 ViT-22B QK-norm (ICML); Ioffe &
        Szegedy 2015 BatchNorm (ICML); Zhang 2023 D2L; Goodfellow 2016.
  Ch 5: Vaswani 2017; Bahdanau 2015 (ICLR); Luong 2015 (EMNLP); Dao 2022 FlashAttention
        (NeurIPS); Dao 2024 FlashAttention-2 (ICLR); Jurafsky & Martin; Zhang 2023 D2L;
        Goodfellow 2016.
  Ch 6: Vaswani 2017; Ainslie 2023 GQA (EMNLP); Chowdhery 2023 PaLM (MQA); Pope 2023;
        Michel 2019 Are Sixteen Heads Really Better than One? (NeurIPS); Voita 2019 (ACL);
        Zhang 2023 D2L; Jurafsky & Martin.
  Ch 7: Vaswani 2017; Dauphin 2017 GLU (ICML); Chowdhery 2023 PaLM (SwiGLU); Elfwing 2018
        SiLU (Neural Networks); Nair & Hinton 2010 ReLU (ICML); Geva 2021 FFN as key-value
        memories (EMNLP); Dai 2022 Knowledge Neurons (ACL); Pope 2023; Zhang 2023 D2L;
        Goodfellow 2016.
  Ch 8: He 2016 ResNet (CVPR); He 2016 Identity Mappings (ECCV); Vaswani 2017; Xiong 2020;
        Liu 2020; Wang 2019; Veit 2016 residual ensembles (NeurIPS); Srivastava 2015 Training
        Very Deep Networks (NeurIPS); Zhang 2023 D2L; Goodfellow 2016.
  Ch 9: Shazeer 2017 sparsely-gated MoE (ICLR); Jacobs 1991 (Neural Computation); Jordan &
        Jacobs 1994 (Neural Computation); Lepikhin 2021 GShard (ICLR); Fedus 2022 Switch
        (JMLR); Dai 2024 DeepSeekMoE (ACL); Du 2022 GLaM (ICML); Zhou 2022 expert choice
        routing (NeurIPS); Rajbhandari 2022 DeepSpeed-MoE (ICML); Clark 2022 routed scaling
        laws (ICML).
  Ch 10: Holtzman 2020 nucleus sampling (ICLR); Fan 2018 top-k (ACL); Bengio 2003; Press &
        Wolf 2017; Ackley 1985 Boltzmann machines/temperature (Cognitive Science); Welleck
        2020 unlikelihood (ICLR); Meister 2023 typical sampling (TACL); Basu 2021 Mirostat
        (ICLR); Goodfellow 2016; Jurafsky & Martin.
  Ch 11: Pope 2023; Kwon 2023 PagedAttention (SOSP); Yu 2022 Orca (OSDI); Leviathan 2023
        speculative decoding (ICML); Williams 2009 Roofline (CACM); Ainslie 2023 GQA;
        Dettmers 2022 LLM.int8() (NeurIPS); Frantar 2023 GPTQ (ICLR); Lin 2024 AWQ (MLSys);
        Agrawal 2024 Sarathi-Serve (OSDI).
  Ch 12: Vaswani 2017; Pope 2023; Brown 2020; Chowdhery 2023 PaLM; Su 2024 RoFormer; Zhang &
        Sennrich 2019 RMSNorm; Ainslie 2023 GQA; Fedus 2022 Switch; Raschka 2024; Jurafsky &
        Martin.

CHAPTER 0 — OVERVIEW
- Intro: the tutorial follows one prompt through the model during inference.
- Diagram 1, pipeline: text → Tokenizer (Ch 1) → Token Embedding (Ch 2) → Decoder Block ×
  n_layers (Ch 3–9) → Final RMSNorm (Ch 4) → LM Head + Sampling (Ch 10) → next token,
  dotted "append, repeat (Ch 11)" back to the tokenizer.
- Diagram 2, inside one block: X [seq, d_model] → RMSNorm → MHA + RoPE → (+) → RMSNorm →
  FFN / MoE → (+) → output [seq, d_model], with both residual connections.
- Chapters table (Ch | Topic | What it covers); every topic in rows 1–12 is a link to its file.
- "## Notation" section: table Symbol | Meaning | Running-example value for every symbol
  above (seq default = 6, the prompt traced in Ch 12; ctx grows by 1 per decode step;
  max_seq = 4,096; n_experts/k = "— (dense model)"; batch = 1). Then the tensor-name list.
- "### seq vs. ctx during generation": first one paragraph explaining the loop: every
  forward pass outputs exactly one new token; prefill feeds the whole prompt and already
  outputs generated token 1; each decode step feeds only the previous pass's token and
  outputs the next; generating n tokens = 1 prefill + (n − 1) decode steps. Then a table for
  a 6-token prompt generating 3 tokens (Step | Tokens fed in | seq | ctx after | Token
  produced): Prefill 6/6 → token 1; Decode 1 1/7 → token 2; Decode 2 1/8 → token 3. Then a
  matching LR diagram ending in "done (3 tokens)". Close with: per-token layers see seq rows;
  only attention reaches back over ctx positions via the KV cache.

CHAPTER 1 — TOKENIZER
- Why sub-words (table: characters / words / sub-words). ~4 characters per token rule of thumb.
- BPE training and encoding. Worked example: corpus hug×10, pug×5, pun×12, bun×4, hugs×5.
  Merges: u+g (20), u+n (16), h+ug (15), with a table of the top pair counts at each step and
  an LR diagram of the merges. Encode "bugs" → [b, ug, s]. Byte-level BPE.
- Vocab table: GPT-2 50,257; Llama-2 32,000 (SentencePiece BPE, byte fallback); Llama-3
  128,256. Special tokens (Llama-2: <unk>=0, <s>=1, </s>=2), the ▁ space marker.
- Pure-Python bpe_encode snippet.

CHAPTER 2 — TOKEN EMBEDDING
- Lookup in E : [vocab, d_model]; equals one-hot × E. Output X : [seq, d_model] = residual
  stream. Tiny example (vocab 4, d_model 3, ids [2, 1, 2]); same token → same vector, so
  position must be added (Ch 3). Size: 131,072,000 params ≈ 262 MB fp16. Weight tying
  (GPT-2 ties, Llama-2 does not). No √d_model scaling in Llama.

CHAPTER 3 — POSITIONAL ENCODING
- Attention alone is order-blind. Absolute: sinusoidal formula, learned [max_seq, d_model]
  table. RoPE: rotate pairs of Q and K (not V) by m·θ_i, θ_i = 10000^(-2i/d_h);
  R(mθ)q · R(nθ)k = q · R((n−m)θ)k → relative position. θ table for d_h=128 (i=0: 1.0,
  i=1: ≈0.866, i=63: ≈0.000115). Note HF "rotate half" layout. NumPy rope().

CHAPTER 4 — NORMALIZATION
- Input/output X : [seq, d_model], normalized per row. LayerNorm and RMSNorm formulas
  (Llama-2 ε = 1e-5). Worked example x = [1, 2, 3, 4]: LayerNorm → [−1.342, −0.447, 0.447,
  1.342]; RMSNorm (mean(x²) = 7.5, rms ≈ 2.739) → [0.365, 0.730, 1.095, 1.461].
  Pre-norm vs post-norm diagrams (Y = X + Sublayer(Norm(X)) vs Y = Norm(X + Sublayer(X))),
  final norm. Param count 266,240.

CHAPTER 5 — SELF-ATTENTION
- Q, K, V projections; d_k = d_h in a real model. softmax(QKᵀ/√d_k + mask)·V with a shape
  table, then a note on decode shapes (Q 1 row, K/V ctx rows from cache, scores [1, ctx]).
  Causal mask grid. Worked example: Q=[[1,0],[1,1]], K=[[1,0],[0,1]], V=[[1,0],[0,10]] →
  weights [[1,0],[0.5,0.5]] → output [[1,0],[0.5,5]]. Cost: ~2·seq·d_model² per projection,
  ~4·seq²·d_model scores (quadratic, FlashAttention), decode ~4·ctx·d_model. NumPy snippet.

CHAPTER 6 — MULTI-HEAD ATTENTION
- Split into n_heads heads of d_h, attend, concat, × W_O. Shape table incl. K/V
  [32, ctx, 128] and scores [seq, ctx]. 4·4096² = 67,108,864 params/layer, ≈ 2.15B total.
  MHA vs GQA vs MQA diagram and table (n_kv_heads; Llama-2-70B 64 Q / 8 KV heads → 8×
  smaller cache). NumPy mha().

CHAPTER 7 — FEED-FORWARD NETWORK
- Y = h + FFN(RMSNorm(h)). Classic FFN(X) = act(X·W1 + b1)·W2 + b2 with a shape table
  listing X [seq, d_model], x = one row, weights, output — plus a highlighted note that the
  input is [seq, d_model] but each row is processed independently with the same weights.
  Diagrams use [seq, d_model] → [seq, d_ff] → [seq, d_model].
- SwiGLU with gate/up/down, shape table, why d_ff ≈ 8/3·d_model (10,922.7 → 11,008).
- Prefill (matrix×matrix, compute-bound) vs decode (vector×matrix, memory-bound); no cache.
- Cost: 135,266,304 params/layer, ≈ 4.33B total (≈ 64%), ≈ 270M FLOPs/token/layer,
  ≈ 8.66 GFLOPs/token, ≈ 270 MB/layer fp16.
- Worked example d_model=2, d_ff=3, ReLU: W1=[[1,0,2],[0,1,−1]], W2=[[1,0],[0,1],[1,1]];
  x=[1,−1] → [1,−1,3] → [1,0,3] → [4,3] → residual [5,2]. Then the same with seq=2,
  X=[[1,−1],[2,1]] → FFN [[4,3],[5,4]] → X+FFN [[5,2],[7,5]]. NumPy for both variants on X.

CHAPTER 8 — RESIDUALS & THE DECODER BLOCK
- h = X + MHA(RMSNorm(X)); Y = h + FFN(RMSNorm(h)). Residual-stream view and diagram.
  Stacking n_layers blocks + final RMSNorm. Params: one block 202,383,360; 32 blocks
  6,476,267,520. Python decoder_block / body using X.

CHAPTER 9 — MIXTURE OF EXPERTS
- MoE replaces the FFN. Takes X [seq, d_model]; router scores X·W_r : [seq, n_experts];
  each row picks its own top-k. Diagram follows one row. Formulas. Worked example
  n_experts=4, k=2, scores [2.0, 1.0, 0.1, −1.0] → gates 0.731 / 0.269. Total vs active
  params: Mixtral 8x7B ≈ 46.7B / ≈ 12.9B (top-2); DeepSeek-V3 ≈ 671B / ≈ 37B (256 routed +
  1 shared, top-8). Shared experts, load balancing, batching, expert parallelism, decode.
  NumPy moe_token (one row) + moe (all rows).

CHAPTER 10 — LM HEAD & SAMPLING
- Diagram starts at final-norm output [seq, d_model] → take last row → h_last [d_model] →
  × W_vocab → logits [vocab] → temperature → top-k/top-p → softmax → sample.
  W_vocab 131M params. Softmax example z=[2,1,0,−1] → [0.644, 0.237, 0.087, 0.032].
  Greedy; temperature table (T=0.5 → [0.865, 0.117, 0.016, 0.002]); top-k=2 → [0.731,
  0.269]; top-p=0.9 → keep 3 → [0.665, 0.245, 0.090]. NumPy sample_next.

CHAPTER 11 — THE INFERENCE LOOP
- Loop diagram: prefill → token 1 → decode one token per pass → stop on EOS/max tokens.
  KV cache with/without diagram. KV bytes = 2 × n_layers × n_kv_heads × d_h × bytes × ctx ×
  batch (define ctx and batch below it): 512 KiB/token, 2 GiB for 4,096 tokens.
  Prefill vs decode table (Tokens per pass (seq); ctx after the pass; bottleneck; TTFT/TPOT).
  Why decode is memory-bound: ≈ 13.5 GB weights per step, ~2 TB/s → ≈ 148 tokens/s upper
  bound at batch 1; prefill of 512 tokens ≈ 6.8 TFLOPs. Speed-up techniques table
  (batching, quantization, GQA/MQA, paged KV cache, speculative decoding). generate() snippet.

CHAPTER 12 — PUTTING IT TOGETHER
- Config line incl. max_seq. Full pipeline diagram. 20-step shape trace for seq = 6
  (prefill, ctx = seq = 6), then the decode note (seq = 1, scores [32, 1, ctx], ctx = 7, 8, …).
  Parameter budget: embedding 131,072,000 (1.9%), attention 2,147,483,648 (31.9%), FFN
  4,328,521,728 (64.2%), norms 266,240, LM head 131,072,000 (1.9%), total 6,738,415,616
  ≈ 6.74B. Per-token: ≈ 13.2 GFLOPs (2 × non-embedding params), extra attention
  ≈ 4·ctx·d_model per layer, 13.5 GB weights read, KV 512 KiB/token, ≈ 148 tokens/s bound.
  Table of variations in other models.
```
