# LLMs from Scratch

Building a large language model end-to-end — tokenization through fine-tuning and deployment — following the curriculum in `curriculum.md`.

The primary reference is **"Build a Large Language Model (From Scratch)"** by Sebastian Raschka (2024), supplemented by foundational papers (Attention Is All You Need, GPT-1/2, LoRA, AdamW, etc.).

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and PyTorch (install separately from [pytorch.org](https://pytorch.org) for the right CUDA/CPU variant).

## Structure

Each lesson is a Jupyter notebook corresponding to a phase in the curriculum. The notebooks are meant to be worked through in order — later lessons build on tokenizer and data-loading code from earlier ones.

- `01_lesson.ipynb` — tokenization from scratch: regex-based splitting, vocab construction, `SimpleTokenizerV1/V2` with `<|unk|>` and `<|endoftext|>` support
- `02_lesson.ipynb` — tokenization on a real corpus: extracts and cleans text from `Introducing-MLOps.pdf` using `pymupdf`, builds a vocab from it, and applies the same tokenizer classes

See `curriculum.md` for the full phase breakdown, milestones, and paper references.
