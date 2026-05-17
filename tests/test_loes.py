import logging

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from loes import select_layers_from_custom_embeddings, select_layers_from_hf, select_layers_from_hf_id


class DummyOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class DummyHFModel(torch.nn.Module):
    def __init__(self, model_type="bert"):
        super().__init__()
        self.proj = torch.nn.Linear(5, 4, bias=False)
        self.config = type("Config", (), {"model_type": model_type, "_name_or_path": f"dummy-{model_type}"})()

    def forward(self, input_ids=None, output_hidden_states=False, **kwargs):
        x = input_ids.float()
        base = self.proj(x).unsqueeze(1).repeat(1, 3, 1)
        hidden_states = (
            base * 0.1,
            base,
            base * 2.0,
            base * 3.0,
        )
        return DummyOutput(hidden_states=hidden_states)


def test_custom_embeddings_classification_returns_k_layers():
    torch.manual_seed(0)
    targets = torch.randint(0, 3, (48,))
    signal = torch.nn.functional.one_hot(targets, num_classes=3).float()
    noisy = torch.randn(48, 6)
    embeddings = torch.stack(
        [
            torch.cat([signal, torch.randn(48, 3)], dim=1),
            torch.cat([noisy, torch.randn(48, 1)], dim=1),
            torch.cat([signal * 0.5, torch.randn(48, 3)], dim=1),
            torch.randn(48, 6),
        ],
        dim=1,
    )

    result = select_layers_from_custom_embeddings(
        embeddings=embeddings,
        targets=targets,
        k=2,
        task="classification",
    )

    assert len(result.selected_layers) == 2
    assert result.num_layers_seen == 4
    assert result.task == "classification"


def test_custom_embeddings_regression_returns_k_layers():
    torch.manual_seed(1)
    x = torch.randn(40, 5)
    y = x[:, 0] * 2.0 - x[:, 1] + 0.1 * torch.randn(40)
    embeddings = torch.stack(
        [
            x,
            torch.randn(40, 5),
            x * 0.25 + 0.1 * torch.randn(40, 5),
        ],
        dim=1,
    )

    result = select_layers_from_custom_embeddings(
        embeddings=embeddings,
        targets=y,
        k=2,
        task="regression",
    )

    assert len(result.selected_layers) == 2
    assert result.task == "regression"


def test_custom_embeddings_label_free_returns_k_layers():
    torch.manual_seed(4)
    base = torch.randn(36, 6)
    embeddings = torch.stack(
        [
            base,
            base + 0.01 * torch.randn(36, 6),
            torch.randn(36, 6),
            torch.randn(36, 6) * torch.linspace(1.0, 3.0, 6),
        ],
        dim=1,
    )

    result = select_layers_from_custom_embeddings(
        embeddings=embeddings,
        k=2,
        task="label_free",
        show_progress=False,
    )

    assert len(result.selected_layers) == 2
    assert len(set(result.selected_layers)) == 2
    assert result.task == "label_free"


def test_custom_embeddings_emits_verbose_logs(caplog):
    torch.manual_seed(3)
    targets = torch.randint(0, 2, (24,))
    signal = torch.nn.functional.one_hot(targets, num_classes=2).float()
    embeddings = torch.stack(
        [
            torch.cat([signal, torch.randn(24, 2)], dim=1),
            torch.randn(24, 4),
            torch.cat([signal * 0.25, torch.randn(24, 2)], dim=1),
        ],
        dim=1,
    )
    logger = logging.getLogger("loes.test")

    with caplog.at_level(logging.INFO, logger="loes.test"):
        result = select_layers_from_custom_embeddings(
            embeddings=embeddings,
            targets=targets,
            k=2,
            task="classification",
            show_progress=False,
            verbose=True,
            logger=logger,
        )

    assert len(result.selected_layers) == 2
    assert "Starting LOES on precomputed embeddings" in caplog.text
    assert "Completed LOES selection" in caplog.text


def test_custom_embeddings_validates_target_length():
    embeddings = torch.randn(8, 2, 3)
    targets = torch.randint(0, 2, (7,))

    with pytest.raises(ValueError, match="targets must contain at least"):
        select_layers_from_custom_embeddings(
            embeddings=embeddings,
            targets=targets,
            k=1,
            task="classification",
            show_progress=False,
        )


def test_invalid_progress_setting_is_rejected():
    embeddings = torch.randn(8, 2, 3)
    targets = torch.randint(0, 2, (8,))

    with pytest.raises(ValueError, match="show_progress"):
        select_layers_from_custom_embeddings(
            embeddings=embeddings,
            targets=targets,
            k=1,
            task="classification",
            show_progress="loud",
        )


def test_hf_path_collects_hidden_states_from_tuple_batches():
    torch.manual_seed(2)
    inputs = torch.randint(0, 5, (16, 5)).float()
    targets = (inputs[:, 0] > 2).long()
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)
    model = DummyHFModel(model_type="bert")

    result = select_layers_from_hf(
        model=model,
        dataset="dummy-text",
        dataloader=loader,
        task="classification",
        k=2,
        max_calibration_samples=12,
    )

    assert len(result.selected_layers) == 2
    assert result.num_calibration_samples == 12
    assert result.pooling == "cls"


def test_hf_path_supports_label_free_tuple_batches():
    torch.manual_seed(5)
    inputs = torch.randint(0, 5, (16, 5)).float()
    loader = DataLoader(TensorDataset(inputs), batch_size=4)
    model = DummyHFModel(model_type="bert")

    result = select_layers_from_hf(
        model=model,
        dataset="dummy-label-free",
        dataloader=loader,
        task="label_free",
        k=2,
        max_calibration_samples=12,
        show_progress=False,
    )

    assert len(result.selected_layers) == 2
    assert result.task == "label_free"
    assert result.num_calibration_samples == 12


def test_hf_id_path_accepts_pytorch_dataloader():
    torch.manual_seed(6)
    inputs = torch.randint(0, 5, (16, 5)).float()
    targets = (inputs[:, 0] > 2).long()
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)

    result = select_layers_from_hf_id(
        "dummy-bert",
        dataloader=loader,
        task="classification",
        k=2,
        model_loader=lambda model_id: DummyHFModel(model_type="bert"),
        max_calibration_samples=12,
        show_progress=False,
    )

    assert len(result.selected_layers) == 2
    assert result.model_name == "dummy-bert"
    assert result.num_calibration_samples == 12


def test_hf_id_path_accepts_hf_style_rows():
    torch.manual_seed(7)
    inputs = torch.randint(0, 5, (16, 5)).float()
    targets = (inputs[:, 0] > 2).long()
    rows = [{"input_ids": inputs[index], "labels": targets[index]} for index in range(inputs.shape[0])]

    result = select_layers_from_hf_id(
        "dummy-bert",
        dataset=rows,
        task="classification",
        k=2,
        model_loader=lambda model_id: DummyHFModel(model_type="bert"),
        batch_size=4,
        max_calibration_samples=12,
        show_progress=False,
    )

    assert len(result.selected_layers) == 2
    assert result.model_name == "dummy-bert"
    assert result.num_calibration_samples == 12
