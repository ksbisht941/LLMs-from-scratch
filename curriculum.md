# LLM From Scratch — Curriculum

**Goal**: Build a large language model from scratch, end-to-end.  
**Prerequisites**: Python, PyTorch, ML/DL experience, RAG/pipeline familiarity.  
**Stack**: Python · PyTorch · numpy · tiktoken · wandb  
**Timeline**: 10–16 weeks (self-paced, ~10 hrs/week)

---

## Phase 1 — Tokenization & Embeddings (1–2 weeks)

**Topics**
- Byte-pair encoding (BPE): merge rules, vocab construction, encode/decode
- WordPiece (conceptual comparison)
- Token embedding layer
- Positional encodings: sinusoidal (original) and learned (GPT-2 style)

**Code milestone**: Implement a BPE tokenizer from scratch in Python (no tiktoken), train it on a small corpus, verify round-trip encoding.

**References**
- Sennrich et al. 2016 — "Neural Machine Translation of Rare Words with Subword Units" (BPE paper)
- Karpathy minbpe: https://github.com/karpathy/minbpe

---

## Phase 2 — Attention & Transformer Core (2–3 weeks)

**Topics**
- Scaled dot-product attention: Q, K, V projections, softmax, scaling by √d_k
- Causal masking (upper-triangular mask)
- Multi-head attention: split heads, parallel attention, concat + output projection
- Feed-forward sublayer: two linear layers with GELU/ReLU
- Residual connections + LayerNorm (pre-norm vs post-norm)
- Assemble one full transformer block

**Code milestone**: Implement a single transformer block in PyTorch. Verify shapes and pass a dummy input through it.

**References**
- Vaswani et al. 2017 — "Attention Is All You Need"
- The Illustrated Transformer (Jay Alammar blog)

---

## Phase 3 — GPT-style Model & Pretraining (2–3 weeks)

**Topics**
- Decoder-only (causal) architecture overview
- Stack N transformer blocks → full GPT class
- Cross-entropy loss on next-token prediction
- Autoregressive generation loop
- Training on tiny-Shakespeare or OpenWebText-small

**Code milestone**: Train a mini-GPT (~10M params) on tiny-Shakespeare. Model should generate coherent Shakespeare-style text.

**References**
- Radford et al. 2018 — "Improving Language Understanding by Generative Pre-Training" (GPT-1)
- Radford et al. 2019 — "Language Models are Unsupervised Multitask Learners" (GPT-2)
- Karpathy nanoGPT: https://github.com/karpathy/nanoGPT

---

## Phase 4 — Training at Scale (2 weeks)

**Topics**
- AdamW optimizer: weight decay decoupling, why it matters for transformers
- Learning rate schedule: linear warmup + cosine decay
- Gradient clipping (global norm clipping)
- Mixed precision training: fp16 / bf16, gradient scaler
- Efficient data loading: packed sequences, DataLoader workers
- Checkpointing and resuming training
- Basic experiment tracking with wandb

**Code milestone**: Re-train mini-GPT with AdamW + cosine schedule + mixed precision. Log loss curves to wandb.

**References**
- Loshchilov & Hutter 2019 — "Decoupled Weight Decay Regularization" (AdamW)
- Chinchilla scaling laws (Hoffmann et al. 2022) — for intuition on compute-optimal training

---

## Phase 5 — Fine-tuning & Alignment (2–3 weeks)

**Topics**
- Supervised fine-tuning (SFT): format instruction datasets, fine-tune on completions only
- LoRA (Low-Rank Adaptation): theory, implement the adapter, freeze base weights
- QLoRA: 4-bit quantization + LoRA for memory efficiency
- RLHF overview: reward model, PPO loop (conceptual)
- DPO (Direct Preference Optimization): simpler alternative to RLHF, implement loss

**Code milestone**: Fine-tune your pretrained mini-GPT on a small instruction dataset using LoRA. Compare outputs before/after.

**References**
- Hu et al. 2021 — "LoRA: Low-Rank Adaptation of Large Language Models"
- Dettmers et al. 2023 — "QLoRA: Efficient Finetuning of Quantized LLMs"
- Rafailov et al. 2023 — "Direct Preference Optimization"
- Ouyang et al. 2022 — "Training language models to follow instructions with human feedback" (InstructGPT/RLHF)

---

## Phase 6 — Inference & Deployment (1–2 weeks)

**Topics**
- KV-cache: implement key/value caching to avoid recomputation
- Quantization: INT8 and INT4 post-training quantization
- Sampling strategies: greedy, temperature, top-k, top-p (nucleus), beam search
- Speculative decoding (conceptual)
- Serving: wrap model in a FastAPI endpoint or explore vLLM
- Evaluation: perplexity, BLEU, BERTScore, human eval basics

**Code milestone**: Add KV-cache to your model. Measure tokens/sec speedup. Serve via a simple FastAPI endpoint.

**References**
- Pope et al. 2022 — "Efficiently Scaling Transformer Inference" (KV-cache analysis)
- Dettmers et al. 2022 — "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"

---

## Capstone Project

Build a small but complete LLM:
1. Train a ~50–100M parameter model on a domain-specific corpus (code, books, Wikipedia slice)
2. Fine-tune with SFT + DPO on a curated instruction dataset
3. Serve it with KV-cache and quantization
4. Evaluate perplexity and generate samples

---

## Book Reference

**"Build a Large Language Model (From Scratch)"** — Sebastian Raschka (2024)  
Covers Phases 1–5 in detail with hands-on PyTorch code. Recommended as the primary textbook alongside the papers above.
