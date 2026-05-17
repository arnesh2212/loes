"""
Examples for running LOES on:
1. An MTEB-style text classification task with BERT.
2. Stanford Cars image classification with DINOv2.

Each example demonstrates both LOES entry points:
- select_layers_from_hf(...)
- select_layers_from_custom_embeddings(...)
"""

from __future__ import annotations

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from loes import select_layers_from_custom_embeddings, select_layers_from_hf


def collect_embeddings_from_hf_model(model, dataloader, pooling="cls", max_samples=None):
    """
    Collect embeddings with shape (n_cal, L, D) from a Hugging Face model.
    This is only for demonstrating the custom-embedding LOES API.
    """

    model.eval()
    device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")
    layer_storage = None
    labels_storage = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"]
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[1:]

            if pooling == "cls":
                pooled = [hidden_state[:, 0, :].cpu() for hidden_state in hidden_states]
            elif pooling == "mean":
                pooled = [hidden_state.mean(dim=1).cpu() for hidden_state in hidden_states]
            else:
                raise ValueError("pooling must be 'cls' or 'mean'")

            if layer_storage is None:
                layer_storage = [[] for _ in pooled]

            for index, layer_tensor in enumerate(pooled):
                layer_storage[index].append(layer_tensor)

            labels_storage.append(labels.cpu())

            if max_samples is not None and sum(chunk.shape[0] for chunk in labels_storage) >= max_samples:
                break

    stacked_labels = torch.cat(labels_storage, dim=0)
    if max_samples is not None:
        stacked_labels = stacked_labels[:max_samples]

    stacked_layers = []
    for per_layer in layer_storage:
        layer_tensor = torch.cat(per_layer, dim=0)
        if max_samples is not None:
            layer_tensor = layer_tensor[:max_samples]
        stacked_layers.append(layer_tensor)

    embeddings = torch.stack(stacked_layers, dim=1)
    return embeddings, stacked_labels


def run_mteb_bert_example() -> None:
    """
    A small MTEB-style example using an Amazon Reviews classification subset.

    MTEB covers many text embedding tasks; this example shows the same LOES usage
    pattern on a text classification dataset with BERT hidden states.
    """

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    dataset = load_dataset("mteb/amazon_reviews_multi", "en", split="train[:128]")

    def collate_fn(batch):
        texts = [row["text"] for row in batch]
        labels = torch.tensor([int(row["label"]) for row in batch], dtype=torch.long)
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded["labels"] = labels
        return encoded

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    result = select_layers_from_hf(
        model=model,
        dataset="mteb/amazon_reviews_multi",
        dataloader=dataloader,
        task="classification",
        k=4,
        max_calibration_samples=128,
    )

    print("MTEB-style BERT example")
    print("Selected layers:", result.selected_layers)
    print("Layer scores:", result.layer_scores)
    print("Pooling:", result.pooling)
    print()

    embeddings, labels = collect_embeddings_from_hf_model(
        model=model,
        dataloader=dataloader,
        pooling="cls",
        max_samples=128,
    )
    custom_result = select_layers_from_custom_embeddings(
        embeddings=embeddings,
        targets=labels,
        k=4,
        task="classification",
    )

    print("MTEB-style BERT example via custom embeddings API")
    print("Selected layers:", custom_result.selected_layers)
    print("Layer scores:", custom_result.layer_scores)
    print()


def run_stanford_cars_dinov2_example() -> None:
    """
    Stanford Cars image classification example with DINOv2.
    """

    model_name = "facebook/dinov2-small"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    dataset = load_dataset("Multimodal-Fatima/StanfordCars_train", split="train[:96]")

    label_key = "label" if "label" in dataset.column_names else "labels"

    def collate_fn(batch):
        pixel_values = processor(
            images=[row["image"] for row in batch],
            return_tensors="pt",
        )["pixel_values"]
        labels = torch.tensor([int(row[label_key]) for row in batch], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    result = select_layers_from_hf(
        model=model,
        dataset="Stanford Cars",
        dataloader=dataloader,
        task="classification",
        k=3,
        max_calibration_samples=96,
    )

    print("Stanford Cars DINOv2 example")
    print("Selected layers:", result.selected_layers)
    print("Layer scores:", result.layer_scores)
    print("Pooling:", result.pooling)
    print()

    embeddings, labels = collect_embeddings_from_hf_model(
        model=model,
        dataloader=dataloader,
        pooling="cls",
        max_samples=96,
    )
    custom_result = select_layers_from_custom_embeddings(
        embeddings=embeddings,
        targets=labels,
        k=3,
        task="classification",
    )

    print("Stanford Cars DINOv2 example via custom embeddings API")
    print("Selected layers:", custom_result.selected_layers)
    print("Layer scores:", custom_result.layer_scores)
    print()


if __name__ == "__main__":
    run_mteb_bert_example()
    run_stanford_cars_dinov2_example()
