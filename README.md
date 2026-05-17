<<<<<<< HEAD
# loes
Code for paper - "Uncovering the Latent Potential of Deep Intermediate Representations" (ICML 26)
=======
# Uncovering the Latent Potential of Deep Intermediate Representations

**LOES** is the reference implementation for **Layerwise Optimal Embedding Selection**, a lightweight module for identifying useful intermediate layers in deep models.

**Accepted at ICML 2026 as a Spotlight.**

[arXiv placeholder](https://arxiv.org/abs/XXXX.XXXXX) | [OpenReview](https://openreview.net/forum?id=6up1qGJwYZ) | [Project page placeholder](https://example.com/loes) | [Code](.)

## Overview

Deep networks expose a stack of intermediate representations, but downstream pipelines often default to the final layer. LOES selects a compact set of layers using a calibration set. It supports supervised selection with labels and label-free selection when labels are unavailable.

LOES works in three practical modes:

1. **Supervised embeddings**: pass `(n_cal, L, D)` embeddings and labels; LOES returns the best layers for classification or regression.
2. **Label-free embeddings**: pass embeddings only; LOES ranks layers using isotropy and redundancy.
3. **Hugging Face model-id mode**: pass a model id plus either a PyTorch dataloader or a Hugging Face-style dataset.

## Repository Structure

```text
loes/
  api.py                 # Core LOES implementation and public API
  __init__.py            # Package exports
tests/
  test_loes.py           # Unit tests for supervised, label-free, and HF-style usage
experiments/
  conf/                  # Experiment configurations
  Results_Final/         # Paper-facing result artifacts
  main_*.py              # Text, image, audio, and segmentation experiment scripts
example_loes.py          # End-to-end usage examples
pyproject.toml           # Package metadata and dependencies
LICENSE                  # MIT license
```

## Installation

```bash
pip install -e .
```

For Hugging Face model-id and dataset loading:

```bash
pip install -e ".[huggingface]"
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Supervised Embeddings

```python
import torch
from loes import select_layers_from_custom_embeddings

embeddings = torch.randn(256, 12, 768)
labels = torch.randint(0, 10, (256,))

result = select_layers_from_custom_embeddings(
    embeddings=embeddings,
    targets=labels,
    k=3,
    task="classification",
    show_progress=True,
    verbose=True,
)

print(result.selected_layers)
print(result.layer_scores)
```

### 2. Label-Free Embeddings

```python
from loes import select_layers_from_custom_embeddings

result = select_layers_from_custom_embeddings(
    embeddings=embeddings,
    k=3,
    task="label_free",
    show_progress=True,
)

print(result.selected_layers)
```

Label-free LOES uses only representation isotropy and redundancy, so labels are not required.

### 3. Hugging Face Model ID With A PyTorch Dataloader

```python
from loes import select_layers_from_hf_id

result = select_layers_from_hf_id(
    "bert-base-uncased",
    dataloader=train_loader,
    task="classification",
    k=4,
    max_calibration_samples=512,
    show_progress=True,
    verbose=True,
)
```

The dataloader can yield `(inputs, targets)` tuples or dictionaries with model inputs and a target key such as `labels`, `label`, `targets`, `target`, or `y`.

### 4. Hugging Face Model ID With A Dataset

```python
from loes import select_layers_from_hf_id

result = select_layers_from_hf_id(
    "bert-base-uncased",
    dataset="ag_news",
    split="train",
    text_key="text",
    target_key="label",
    task="classification",
    k=4,
    max_calibration_samples=512,
)
```

If the dataset is already tokenized, LOES will collate keys such as `input_ids`, `attention_mask`, `pixel_values`, or `input_values`. For raw text datasets, it uses the model tokenizer and a text column such as `text`, `sentence`, `query`, or `document`.

## Public API

```python
from loes import (
    select_layers_from_custom_embeddings,
    select_layers_from_hf,
    select_layers_from_hf_id,
)
```

All selectors return:

```python
LOESResult(
    selected_layers=[...],
    layer_scores=[...],
    task="classification",
    num_layers_seen=12,
    num_calibration_samples=256,
    pooling="cls",
    dataset="...",
    model_name="...",
)
```

## Progress Bars And Logs

LOES uses `tqdm` progress bars and Python's standard `logging` module.

```python
result = select_layers_from_custom_embeddings(
    embeddings,
    targets=labels,
    k=4,
    task="classification",
    show_progress="auto",
    verbose=True,
)
```

`show_progress="auto"` is the default and displays bars only in interactive terminals. Use `show_progress=True` or `"on"` to force bars, and `show_progress=False` or `"off"` for quiet scripts.

`verbose=True` emits logs for calibration collection, layer scoring, greedy layer selection, and final selected layers.

## Pooling Rules

When `pooling="auto"`:

- Text encoder families such as BERT, RoBERTa, DeBERTa, DistilBERT, and ModernBERT use `cls`.
- Vision transformer families such as ViT, DeiT, BEiT, DINOv2, and DINOv3 use `cls`.
- Audio encoders such as Wav2Vec2, HuBERT, WavLM, Whisper, and AST use `mean`.
- Unknown model types fall back to `mean`.

You can also force `pooling="cls"`, `pooling="mean"`, or `pooling="masked_mean"`.

## Task Semantics

- `task="classification"`: labels can be integer class ids or one-hot floating targets.
- `task="regression"`: targets can be shaped as `(n_cal,)` or `(n_cal, out_dim)`.
- `task="label_free"`: targets are optional and ignored; selection uses isotropy and redundancy only.

## Experiments

Experiment scripts and configuration files live under `experiments/`. Final result artifacts used for paper tables and plots are stored under:

```text
experiments/Results_Final/
```

Typical entry points include:

```bash
python experiments/main_text.py
python experiments/main_image.py
python experiments/main_audio.py
python experiments/main_image_seg.py
```

## Development

```bash
python -m pytest -q
python -m compileall -q loes tests
```

## Citation

```bibtex
@inproceedings{
anonymous2026uncovering,
title={Uncovering the Latent Potential of Deep Intermediate Representations},
author={Anonymous},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=6up1qGJwYZ}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
>>>>>>> 36b7855 (Initial Code)
