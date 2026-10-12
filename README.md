# irishQwen

Classroom version of Qwen3 for CSE 10124 — Building ChatGPT.

An educational implementation using PyTorch tensor operations and explicit forward/backward methods. It includes grouped-query attention with QK normalization, rotary position embeddings, RMSNorm, SwiGLU, optional sparse mixture-of-experts blocks, and KV-cached generation.

## Setup

Use Python 3.10 or newer. Run commands from the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The model selects CUDA, Apple MPS, or CPU according to availability.

## Train on Shakespeare

The teaching corpus and its vocabulary are included in `Datasets/`:

```sh
python train_qwen_shakespeare.py --optimizer adam
python train_qwen_shakespeare.py --help
```

Training saves its checkpoint to `Datasets/qwen_shakespeare.pt` by default. Generated model checkpoints are ignored by Git.

## Import pretrained Qwen3 weights

Install the additional conversion dependency, then download and convert the default Qwen3-0.6B model:

```sh
pip install -r requirements-conversion.txt
python scripts/import_qwen3.py
```

This downloads model weights from Hugging Face and creates a roughly 2.2 GB float32 checkpoint plus tokenizer assets in `Datasets/`. Allow sufficient disk space and memory for both the original and converted weights. These generated assets are not included in the repository.

```python
from irishGPT.irishQwen_kv import IrishQwenKV
from irishGPT.qwen_tokenizer import QwenTokenizer

checkpoint = "Datasets/qwen3_0.6b_converted.pt"
tokenizer = QwenTokenizer("Datasets/qwen3_tokenizer_assets/tokenizer.json")
model = IrishQwenKV.from_checkpoint_config(checkpoint)
model.load_converted_qwen3_checkpoint(checkpoint)
prompt = tokenizer.encode(tokenizer.chat_prompt("What is a transformer?"))
tokens = model.chat(prompt, max_new_tokens=100, **model.NON_THINKING_SAMPLING)
print(tokenizer.decode(tokens))
```

## Code map

- `irishGPT/irishQwen.py`: model, manual training operations, checkpoints, and sampling.
- `irishGPT/irishQwen_kv.py`: cached inference with prefill and decode.
- `irishGPT/qwen_transformer*.py`: dense/MoE decoder blocks and cached attention.
- `irishGPT/grouped_query_attention.py`, `rotary_embedding.py`, `rms_norm.py`, `swiglu.py`, and `qwen_moe.py`: architectural building blocks.
- `irishGPT/qwen_tokenizer.py`: byte-level BPE tokenizer and chat prompt formatting.
- `scripts/import_qwen3.py`: Hugging Face checkpoint conversion.
- `train_qwen_shakespeare.py`: small-model training example.

The `irishGPT` package name and shared GPT teaching modules are retained because the Qwen implementation inherits training behavior from `IrishChat` and reuses those components. Source files were copied unchanged from the course implementation.
