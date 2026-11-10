# CS781 Capstone: In-House Legal Intelligence System

Code for my CS781 (IIT Kanpur) capstone project. A law firm that cannot send client documents to a cloud API needs language models it can host itself. I pretrained four small legal language models from scratch on the course corpus, then adapted them to six downstream legal tasks.

- **Author:** Dhruv Kejriwal (Roll No. 240357), Codabench user `d_240357`
- **Hardware:** Kaggle, 2 × NVIDIA T4 (16 GB each, fp16)
- **Data:** only the official [`Exploration-Lab/CS-781-Capstone`](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone) dataset
- **Rules I followed:** every model and tokenizer is written from scratch in PyTorch; no pretrained weights are loaded anywhere, not even for evaluation

No model weights, checkpoints or data are stored here. The notebooks download the Phase-1 checkpoints from my Hugging Face account ([`D-240357`](https://huggingface.co/D-240357)) and the data from the official dataset.

---

## Contents

1. [Results at a glance](#results-at-a-glance)
2. [Repository layout](#repository-layout)
3. [How to run](#how-to-run)
4. [Phase 1: four foundation models](#phase-1-four-foundation-models)
5. [Shared fine-tuning framework (Phases 2–7)](#shared-fine-tuning-framework-phases-27)
6. [Phase 2: Legal NER](#phase-2-legal-named-entity-recognition)
7. [Phase 3: Rhetorical roles](#phase-3-rhetorical-role-prediction)
8. [Phase 4: Judgment prediction](#phase-4-court-judgment-prediction)
9. [Phase 5: Statute identification](#phase-5-legal-statute-identification)
10. [Phase 6: Summarization](#phase-6-abstractive-summarization)
11. [Phase 7: Prior case retrieval](#phase-7-prior-case-retrieval)
12. [Codabench submissions](#codabench-submissions)
13. [What failed and what I learned](#what-failed-and-what-i-learned)

---

## Results at a glance

Best official Codabench score per phase (test set), rank among all entrants, and the model behind it.

| Phase | Task | Metric | Best model | Local dev | Codabench | Rank |
|---|---|---|---|---|---|---|
| 1 | Legal LM pretraining | BERTScore F1 | A3 Llama-style, 600M tokens | ROUGE-L 0.331 | **0.8315** | 68 / 142 |
| 2 | Legal NER | strict macro-F1 | A3, partial fine-tuning | 0.487 | **0.4964** | 83 / 132 |
| 3 | Rhetorical roles | macro-F1 | A4 MoE, LoRMA | 0.578 | **0.5449** | 30 / 131 |
| 4 | Judgment prediction | macro-F1 | A3, distillation (KD) | 0.711 | **0.7122** | 73 / 131 |
| 5 | Statute identification | macro-F1 | A3, partial fine-tuning | 0.379 | **0.3624** | 81 / 131 |
| 6 | Summarization | BERTScore F1 | A4 Houlsby adapters | ROUGE-L 0.229 | **0.8337** | 83 / 125 |
| 7 | Prior case retrieval | micro-F1@7 | A3, LoRMA + BM25 fusion | 0.351 | **0.3219** | 94 / 130 |

Ranks were recorded on 9 November 2026. In total I made 38 valid submissions (5, 4, 6, 6, 6, 5, 6 per phase). The full report, with data analysis, ablations and error analysis, is submitted separately.

**Main findings**
- The Llama-style decoder (A3) trained on 600M tokens was the best backbone in almost every phase. The number of pretraining tokens mattered more than the architecture.
- Lifting the causal mask turns a decoder into a usable encoder for labelling tasks.
- LoRMA came within 0.015 of full fine-tuning, or beat it, on the two strongest backbones. On the weakest one it was the most fragile method.
- With the same training time, a distilled model with half the layers beat the full model it learned from in 15 of 20 classification and retrieval cases.
- Cheap decision rules (constrained decoding, tuned thresholds, score fusion) moved scores as much as switching fine-tuning methods did.

---

## Repository layout

```
phase1/
  arch1_gpt2/              A1: GPT-2 style decoder, word-level tokenizer (notebook)
  arch2_encoder_decoder/   A2: T5-style encoder-decoder
  arch3_llama/             A3: Llama-style decoder  (best Phase-1 model)
  arch4_moe/               A4: DeepSeek-style mixture-of-experts decoder
phase2/ … phase7/
  src/common/              task code shared by all four backbones
  src/archN/               backbone adapters for each Phase-1 model
  src/build_notebooks.py   script that generated the Kaggle notebooks
  notebooks/               one Kaggle notebook per backbone (phaseP_archN_<task>.ipynb)
```

Each Phase-1 folder has `bpe_tokenizer.py` (byte-level BPE), `prep_data.py` (tokenize the corpus into a memory-mapped file), `model.py`, `train.py`, `config.py` (all hyperparameters) and the notebook. Backbone adapters for Phases 4, 5 and 7 are the ones in `phase3/src/archN/`.

**Notebooks with full Kaggle run logs:**
- all four Phase-1 notebooks; for A4, the log is from the first 100M-token run, `phase1_arch4_moe_run1_100M_tokens.ipynb`
- all four Phase-2 notebooks
- the best model of Phases 3–7: `phase3_arch4_rr`, `phase4_arch3_cjpe`, `phase5_arch3_lsi`, `phase6_arch3_sum`, `phase6_arch4_sum`, `phase7_arch3_pcr`

The other notebooks are the generated versions without output. They contain exactly the same code.

---

## How to run

Every notebook is self-contained: its first cells write the source files to disk (`%%writefile`), then run them.

1. Create a Kaggle notebook with **GPU T4 × 2** and import the `.ipynb` file.
2. Data: Phase 1 downloads `Exploration-Lab/CS-781-Capstone` from Hugging Face. Phases 2–7 look for it under `/kaggle/input` and otherwise download it from Hugging Face.
3. Add a Kaggle secret named `HF` containing a Hugging Face token with access to the dataset (and write access if you want checkpoints pushed).
4. Run all cells. Each notebook:
   - checks the GPUs and the parameter budget
   - prepares the data
   - trains every method, one experiment per GPU in parallel
   - selects checkpoints and decision rules on dev
   - writes one Codabench-ready `submission.zip` per method, plus the best one

**Time budgets:** Phase 1 runs for about 5–12 hours per model and resumes automatically from Hugging Face if the session dies. Phases 2–7 take about 1–3 hours per notebook (every method has a fixed wall-clock budget).

**Settings:** all hyperparameters are in the notebook's config cell (`config.py`). For a quick CPU smoke test, set the `*_CFG_OVERRIDE` environment variable or the `*_LIMIT` options shown in the config.

**Requirements:** Python 3.10+, PyTorch 2.x, NumPy, pandas, `huggingface_hub`. No tokenizer or model libraries are used.

---

## Phase 1: four foundation models

The course asked for four substantially different architectures (50M–300M parameters). I varied one thing at a time against a common recipe: AdamW (β = 0.9, 0.95; weight decay 0.1), warm-up then cosine decay to 10%, gradient clipping 1.0, 64K tokens per optimiser step, next-token cross-entropy, fp16 autocast, and DDP across both T4s.

| Model | Architecture | Params | Tokens | Val loss |
|---|---|---|---|---|
| A1 GPT-2 | 12 layers, d=768, learned positions, LayerNorm, GELU MLP (3072), word vocab, context 2048 | 111.2M | 600M | n/a (word-level) |
| A2 Encoder-decoder | 6+6 layers, T5-style, RoPE, RMSNorm, GQA 12/4, SwiGLU; encoder reads 1,024 tokens, decoder writes 320 | 109.5M | ~150M | 3.45 |
| **A3 Llama-style** | 12 layers, d=768, RoPE, RMSNorm, GQA 12/4, SwiGLU (2048), tied embeddings, z-loss 1e-4, context 2048 | **100.1M** | **600M** | **2.213** |
| A4 MoE (DeepSeek-style) | 12 layers, multi-head latent attention (KV latent 192 + 16 RoPE dims), 1 dense + 11 MoE layers (32 routed experts, top-4, + 1 shared), auxiliary-loss-free balancing, multi-token prediction | 222.6M (81.2M active) | 311M | 2.394 |

**Tokenizers**
- **A1:** a word-level tokenizer (32K most frequent lower-cased words). It is fast but lossy: 2.6% `<unk>` on NER documents.
- **A2–A4:** a byte-level BPE written from scratch with the standard library only. It uses a GPT-2-like regex on plain `re`, a 32,000-entry vocabulary (31,742 merges learned on 40 MB per source) and no `<unk>`, so citations and names round-trip exactly. Four worker processes encode about 3.5M tokens/s into a `uint16` memory-mapped file. A2 and A4 reuse A3's tokenizer and token files.

**Data mix:** the four corpus sources (Acts, High Courts, Supreme Court, Tribunals) are interleaved with equal weight by characters. Every 100th document is held out for validation.

**Engineering for A3:**
- fused QKV
- PyTorch SDPA attention
- `torch.compile`
- RMSNorm statistics in fp32
- z-loss to keep fp16 logits stable
- checkpoints, optimiser state and step pushed to Hugging Face every 250 steps for exact resumption

A3 trained 9,155 steps in 5.3 hours at about 31.8K tokens/s. On 48 held-out documents (512-word prompt, next 128 words) it scores ROUGE-L 0.331 and BLEU-4 0.208. For submission it samples with temperature 0.7, top-k 50, top-p 0.92 and a KV cache, and the output is cut to 128 words.

**Engineering for A4:** tokens are dispatched through fixed-capacity expert buffers (capacity factor 1.25) with one batched matmul, so all shapes are static and `torch.compile` can fuse them. The token-drop rate fell from 11% to 4.8% during training. Per token, A4 learns about as fast as A3, but at half the throughput on a T4.

---

## Shared fine-tuning framework (Phases 2–7)

Every downstream model starts from a Phase-1 checkpoint. Each notebook:
- loads a backbone and adds a small task head
- trains **six methods on the same wall-clock budget**, with one experiment per GPU
- picks the checkpoint and decision rule on dev
- writes a validated submission zip

The validation checks that `prediction.json` is at the root, that all IDs are present with no extras, and that the types are correct.

**Decoders as encoders.** For every task except summarization, the causal mask is lifted during fine-tuning so attention runs both ways (as in LLM2Vec). A2 uses only its encoder.

**Methods.** These are implemented in `src/common/*_core.py` by wrapping `nn.Linear` layers; the MoE router is never wrapped.

| Method | What is trained |
|---|---|
| full | everything (with layer-wise LR decay) |
| **LoRMA** (compulsory) | multiplicative low-rank update `h = (α/r · BA + I) W₀x`, r = 8, α = 8, B initialised to 0 |
| LoRA | additive low-rank update, r = 8, α = 16 |
| Houlsby adapters | bottleneck 64 after attention / FFN |
| IA³ | learned rescaling vectors |
| BitFit | biases only |
| prompt tuning | 16 learned input vectors |
| partial | top 4 of 12 layers (2 of 6 for A2) |
| gradual unfreezing | layers unfrozen top-down with LR decay |
| **KD (distillation)** | a half-size copy (every second layer) learns from the true labels and from the full model's softened probabilities (T = 2, λ = 0.5), plus a cosine loss on sentence or document vectors |

Shared settings: AdamW (weight decay 0.01), warm-up then decay to 10%, clipping 1.0, head LR 1e-3, adapter LR 5e-4, seed 13.

---

## Phase 2: Legal Named Entity Recognition

**Task:** character spans for 12 entity types in full judgments. Scored by strict macro-F1, so a span off by one character is wrong.

**Approach**
- **Exact offsets.** Text is split into pieces with recorded offsets (runs of letters, runs of digits, single punctuation marks). Each piece is tokenized separately, and predicted spans map back to exact characters. Round-trip check on the gold labels: 0.994 strict macro-F1.
- **Windows.** Each judgment is read in 512-token windows with stride 384. Each token keeps its label from the window where it is most central.
- **Decoding.** A linear head predicts BIO tags. A constrained Viterbi decoder forbids impossible transitions, which gives +0.04 to +0.14 over greedy decoding.
- **Training.** 10 epochs, patience 3. Dev = 8 held-out training judgments (Phase 2 has no dev split).

**Methods:** full, LoRMA, LoRA, partial, KD. **Best:** A3 partial, 0.487 dev, **0.4964** Codabench.

## Phase 3: Rhetorical role prediction

**Task:** one of 13 rhetorical roles for every sentence of Corporate Law and Income Tax judgments (macro-F1).

**Approach (hierarchical)**
- Sentences are packed into 512-token sequences so each is read in context.
- Each sentence is mean-pooled; position and domain embeddings are added; a BiLSTM runs over the document, with a CRF on top.
- A class-weighted loss and an auxiliary "does the role change here?" task help the rare roles.
- A prior correction τ and CRF vs. argmax decoding are picked on dev. The CRF won in 21 of 24 runs.

**Methods:** full, LoRMA, LoRA, IA³, prompt, KD. **Best:** A4 LoRMA, 0.578 dev, **0.5449** Codabench.

## Phase 4: Court judgment prediction

**Task:** predict whether an appeal was accepted, from a judgment with the decision removed (macro-F1).

**Approach**
- **Input.** Each case is the opening 512 tokens plus the last three 512-token chunks, combined by attention pooling.
- **Imbalance.** The training split is 59/41 but dev and test are 50/50, so the loss is class-balanced and the decision threshold is tuned on dev (up to +0.23 for weak models).
- **Leak.** The public `test.json` contains a `label` field, and no code reads it.
- **Budget.** 35 minutes per run.

**Methods:** full, LoRMA, BitFit, Houlsby, gradual unfreezing, KD. **Best:** A3 KD, 0.711 dev, **0.7122** Codabench.

## Phase 5: Legal statute identification

**Task:** multi-label prediction over 100 statutes (macro-F1).

**Approach**
- **Input.** Up to six evenly spread 512-token chunks per document.
- **Head.** Label-wise attention (CAML/LWAN): each statute attends to its own evidence, followed by independent sigmoids.
- **Loss.** Asymmetric loss (γ⁻ = 2, clip 0.05).
- **Thresholds.** One per statute, tuned on dev; this beat a single global threshold in every run (+0.03 to +0.06). Every document gets at least one label.
- **Budget.** 40 minutes per run.

**Methods:** full, LoRMA, Houlsby, partial, prompt, KD. **Best:** A3 partial, 0.379 dev, **0.3624** Codabench.

## Phase 6: Abstractive summarization

**Task:** abstractive summary of a judgment (BERTScore F1).

**Approach**
- **Model.** Decoders are trained as prefix language models on `[source] <eos> [summary]` (loss on the summary only); A2 uses its encoder-decoder.
- **Source.** 1,000 tokens: 40% from the opening plus three evenly spaced pieces of the rest.
- **Decoding.** Greedy with a KV cache, repetition penalty 1.15 and no repeated 4-grams.
- **Length.** Document words / 4.6 (the median compression in the training data), clipped to 150–900 words.
- **Selection.** BERTScore needs pretrained weights, so models were selected by ROUGE on 40 held-out training documents.
- **Budget.** 40 minutes per run.

**Methods:** full, LoRMA, LoRA, Houlsby, prompt, KD. **Best:** A3 Houlsby on dev (ROUGE-L 0.234). Its upload returned no score, so the best official score, **0.8337**, came from A4 Houlsby.

## Phase 7: Prior case retrieval

**Task:** for each query case, return the 7 most likely prior cases from a candidate pool (micro-F1@7).

**Approach (hybrid)**
- **BM25** is written from scratch with an inverted index and uses only the query's 300 most informative terms. This raised BM25 alone from 0.135 to 0.241.
- **Dense bi-encoder.** Queries and candidates share one encoder: three 512-token windows, mean-pooled, projected to a normalised 256-d vector. It is trained with InfoNCE (τ = 0.05), with in-batch negatives and hard negatives from BM25's top 30. Co-relevant candidates and the query's own copy are masked out.
- **Fusion.** Final score = α · z(dense) + (1 − α) · z(BM25), with α tuned on dev. The query itself is never returned.
- **Budget.** 30 minutes per run.

**Methods:** full, LoRMA, partial, IA³, prompt, KD. **Best:** A3 LoRMA (dense 0.278, BM25 0.241, fused 0.351 dev), **0.3219** Codabench.

---

## Codabench submissions

Every scored submission and its official primary metric. In Phases 3–7 the six submissions were chosen by one rule: the best run of each backbone, then LoRMA and KD from the strongest backbone, then the next-best distinct methods.

| Phase | Submissions (official score) |
|---|---|
| 1 | A3 0.8315 · A4 0.8273 · A2 0.8184 · A1 0.8152 · re-submission 0.8038 |
| 2 | A3 partial 0.4964 · A4 LoRA 0.4523 · A2 KD 0.2454 · A1 KD 0.2267 |
| 3 | A4 LoRMA 0.5449 · A3 full 0.5198 · A4 LoRA 0.4992 · A4 full 0.4415 · A1 KD 0.4162 · A2 KD 0.3566 |
| 4 | A3 KD 0.7122 · A3 gradual unfreezing 0.7059 · A1 KD 0.6585 · A3 LoRMA 0.6554 · A4 KD 0.6420 · A2 KD 0.6260 |
| 5 | A3 partial 0.3624 · A3 full 0.3497 · A3 LoRMA 0.3305 · A1 KD 0.3302 · A4 KD 0.2926 · A2 KD 0.2184 |
| 6 | A4 Houlsby 0.8337 · A3 LoRMA 0.8336 · A1 Houlsby 0.8298 · A3 KD 0.8272 · A2 full 0.8104 |
| 7 | A3 LoRMA 0.3219 · A3 IA³ 0.3166 · A3 partial 0.3160 · A1 LoRMA 0.3071 · A4 KD 0.3012 · A2 KD 0.2658 |

---

## What failed and what I learned

- **A2's first version memorised its data.** Its validation loss rose from 3.1 to 3.7 while training loss fell to 1.5, because the streaming loader cycled over a small slice of data. I rebuilt it on pre-tokenized files with a document-level split.
- **A1's first resume broke the learning-rate schedule.** I rebuilt the schedule from the saved step count.
- **A4 stopped at 311M of a planned 600M tokens** when the Kaggle session ended, because the MoE runs at half the dense model's speed.
- **A4 needed two fixes before it would run:** an Inductor crash on the expert one-hot, and running out of memory at micro-batch 4.
- **A2's NER LoRMA and partial runs collapsed** to predicting no entities. A class-weighted loss would probably have prevented this.
- **Wrapping the MoE router with BitFit broke it**, so the wrapper now skips the router.
- **An early Phase-3 zip was rejected** because it contained a folder at its root. Every zip is now validated automatically.
- **The Phase-4 dev file was replaced by the organisers mid-phase**, so A1's Phase-4 runs used the old 3,505-case dev set and A2–A4 the new 1,988-case one.

## LLM usage

Claude (Anthropic) and Gemini (Google) were used during this project. The full declaration of how they were used is part of the submitted report.

## References

The main papers this code builds on:
- LoRMA (Bihany et al., 2025)
- LoRA (Hu et al., 2022)
- Houlsby adapters (2019)
- IA³ (Liu et al., 2022)
- BitFit (Ben Zaken et al., 2022)
- Prompt tuning (Lester et al., 2021)
- ULMFiT (Howard & Ruder, 2018)
- Knowledge distillation (Hinton et al., 2015)
- DeepSeek-V2/V3 and DeepSeekMoE
- Auxiliary-loss-free balancing (Wang et al., 2024)
- RoFormer, RMSNorm, GQA, SwiGLU, LLM2Vec
- CRF (Lafferty et al., 2001)
- CAML (Mullenbach et al., 2018)
- Asymmetric loss (Ridnik et al., 2021)
- BM25 (Robertson & Zaragoza, 2009)
- DPR (Karpukhin et al., 2020)
- InfoNCE (van den Oord et al., 2018)
- ANCE (Xiong et al., 2021)
- Fusion for hybrid retrieval (Bruch et al., 2023)
- IL-TUR (Joshi et al., 2024)

The full bibliography is in the report.
