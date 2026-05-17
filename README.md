<div align="center">

<br>

<p>
  <img alt="ICML 2026 Spotlight" src="https://img.shields.io/badge/ICML%202026%20Spotlight-accepted-6f42c1?style=for-the-badge">
</p>


<h1>
  Uncovering the Latent Potential of<br>
  Deep Intermediate Representations
</h1>

<h3>
  Layerwise Optimal Embedding Selection for supervised and label-free representation discovery
</h3>

<p>
  <strong>Arnesh Batra</strong><sup>1</sup> ·
  <strong>Arush Gumber</strong><sup>*1</sup> ·
  <strong>Aniket Khandelwal</strong><sup>*1</sup> ·
  <strong>Jashn Khemani</strong><sup>1</sup> ·
  <strong>Anubha Gupta</strong><sup>1</sup>
</p>

<p>
  <sup>1</sup>SBILab, Indraprastha Institute of Information Technology Delhi, Delhi, India<br>
  <sup>*</sup>Equal contribution
</p>

<p>
  <a href="#"><img alt="arXiv coming soon" src="https://img.shields.io/badge/arXiv-coming%20soon-b31b1b?style=flat-square&logo=arxiv&logoColor=white"></a>
  <a href="https://openreview.net/forum?id=6up1qGJwYZ"><img alt="OpenReview" src="https://img.shields.io/badge/OpenReview-paper-8c1d40?style=flat-square"></a>
  <a href="https://example.com/loes"><img alt="Project Page" src="https://img.shields.io/badge/Project-page-2f80ed?style=flat-square"></a>
  <a href="."><img alt="Code" src="https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github"></a>
  <a href="LICENSE"><img alt="License MIT" src="https://img.shields.io/badge/License-MIT-green?style=flat-square"></a>
</p>

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-3776ab?style=flat-square&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?style=flat-square&logo=pytorch&logoColor=white">
  <img alt="Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-compatible-ffcc4d?style=flat-square">
</p>

<br>

</div>

---

**LOES** is the reference implementation for **Layerwise Optimal Embedding Selection**, a lightweight module for identifying useful intermediate layers in deep models.

LOES supports supervised layer selection with labels and label-free selection when labels are unavailable.

## Highlights

- **Supervised embeddings**: pass `(n_cal, L, D)` embeddings and labels; LOES returns the best layers for classification or regression.
- **Label-free embeddings**: pass embeddings only; LOES ranks layers using isotropy and redundancy.
- **Hugging Face model-id mode**: pass a model id plus either a PyTorch dataloader or a Hugging Face-style dataset.
- **Progress and logs**: use `show_progress=True` and `verbose=True` for clean progress bars and structured run logs.

## Installation

```bash
pip install -e .
```

For Hugging Face model-id and dataset loading:

```bash
pip install -e ".[huggingface]"
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
    dataset=train_loader,  # dataloader=train_loader also works
    task="classification",
    k=4,
    max_calibration_samples=512,
    show_progress=True,
    verbose=True,
)
```

The dataloader can yield `(inputs, targets)` tuples or dictionaries with model inputs and a target key such as `labels`, `label`, `targets`, `target`, or `y`.

### 4. Hugging Face Model ID With A Hugging Face Dataset

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

`dataset` can be a Hugging Face dataset id, a loaded `Dataset`, or a `DatasetDict`; for `DatasetDict`, LOES uses the requested `split`. If the dataset is already tokenized, LOES will collate keys such as `input_ids`, `attention_mask`, `pixel_values`, or `input_values`. For raw text datasets, it uses the model tokenizer and a text column such as `text`, `sentence`, `query`, or `document`.

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
