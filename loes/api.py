from __future__ import annotations

from dataclasses import dataclass
import logging
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


Batch = Union[Tuple[Any, Any], Dict[str, Any]]
TensorLike = Union[torch.Tensor, Sequence[torch.Tensor]]
ProgressSetting = Union[bool, str]
ModelLoader = Callable[[str], torch.nn.Module]


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


TEXT_CLS_MODEL_TYPES = {
    "albert",
    "bert",
    "camembert",
    "deberta",
    "deberta-v2",
    "distilbert",
    "electra",
    "ernie",
    "modernbert",
    "roberta",
    "xlm-roberta",
}
IMAGE_CLS_MODEL_TYPES = {
    "beit",
    "deit",
    "dinov2",
    "dinov3_convnext",
    "dinov3_vit",
    "vit",
}
AUDIO_MEAN_MODEL_TYPES = {
    "audio-spectrogram-transformer",
    "hubert",
    "sew",
    "unispeech",
    "unispeech-sat",
    "wav2vec2",
    "wavlm",
    "whisper",
}
LABEL_FREE_TASKS = {"label_free", "label-free", "unsupervised"}
TARGET_KEYS = ("labels", "label", "targets", "target", "y")
MODEL_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "position_ids",
    "pixel_values",
    "input_values",
)
TEXT_KEYS = ("text", "sentence", "sentence1", "query", "document", "content", "review")


@dataclass
class LOESResult:
    selected_layers: List[int]
    layer_scores: List[float]
    task: str
    num_layers_seen: int
    num_calibration_samples: int
    pooling: str
    dataset: Optional[Any] = None
    model_name: Optional[str] = None


def _log_info(logger: logging.Logger, verbose: bool, message: str, *args: Any) -> None:
    if verbose:
        logger.info(message, *args)


def _normalize_task(task: str) -> str:
    if task in {"classification", "regression"}:
        return task
    if task in LABEL_FREE_TASKS:
        return "label_free"
    raise ValueError("task must be one of: 'classification', 'regression', 'label_free'")


def _safe_len(value: Any) -> Optional[int]:
    try:
        return len(value)
    except TypeError:
        return None


def _progress_is_enabled(show_progress: ProgressSetting) -> bool:
    if isinstance(show_progress, str):
        normalized = show_progress.lower()
        if normalized not in {"auto", "on", "off"}:
            raise ValueError("show_progress must be a bool or one of: 'auto', 'on', 'off'")
        if normalized == "auto":
            return sys.stderr.isatty()
        return normalized == "on"
    return bool(show_progress)


def _iter_with_progress(
    iterable: Iterable[Any],
    *,
    show_progress: ProgressSetting,
    desc: str,
    total: Optional[int] = None,
    unit: str = "it",
    logger: logging.Logger = LOGGER,
    verbose: bool = False,
) -> Iterable[Any]:
    if not _progress_is_enabled(show_progress):
        return iterable

    try:
        from tqdm.auto import tqdm
    except ImportError:
        if verbose:
            logger.warning(
                "Progress was requested for '%s', but tqdm is not installed. "
                "Install tqdm or use show_progress=False.",
                desc,
            )
        return iterable

    return tqdm(iterable, total=total, desc=desc, unit=unit, leave=False)


def compute_isotropy(x: torch.Tensor, eps: float = 1e-6) -> float:
    x_centered = x - x.mean(dim=0, keepdim=True)
    n_samples, n_features = x_centered.shape

    if n_features > 512:
        singular_values = torch.linalg.svdvals(x_centered)
        eigenvalues = (singular_values.square() / max(n_samples, 1)).clamp(min=eps)
    else:
        cov = (x_centered.T @ x_centered) / max(n_samples, 1)
        cov = cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
        eigenvalues = torch.linalg.eigvalsh(cov).real.clamp(min=eps)

    return (eigenvalues.mean() / (eigenvalues.std(unbiased=False) + eps)).item()


def closed_form_ridge(x: torch.Tensor, y: torch.Tensor, reg: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)
    eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    weights = torch.linalg.solve(x_centered.T @ x_centered + reg * eye, x_centered.T @ y_centered)
    bias = (y.mean(dim=0, keepdim=True) - x.mean(dim=0, keepdim=True) @ weights).squeeze(0)
    return weights, bias


def _as_float_targets(targets: torch.Tensor, task: str) -> torch.Tensor:
    task = _normalize_task(task)
    if task == "label_free":
        raise ValueError("label_free selection does not use targets")

    if task == "classification":
        if targets.ndim == 2 and torch.is_floating_point(targets):
            return targets.float()
        if targets.ndim != 1:
            raise ValueError("Classification targets must be class indices or a one-hot matrix.")
        if not torch.is_floating_point(targets):
            targets = targets.long()
        return F.one_hot(targets.long(), num_classes=int(targets.max().item()) + 1).float()

    if targets.ndim == 1:
        return targets.float().unsqueeze(-1)
    return targets.float()


def _triangle_score(features: torch.Tensor, labels: torch.Tensor, max_triplets: int = 200) -> float:
    unique_labels = torch.unique(labels)
    if unique_labels.numel() < 3:
        return 0.0

    centroids = torch.stack([features[labels == label].mean(dim=0) for label in unique_labels])
    if centroids.shape[0] < 3:
        return 0.0

    triplet_count = min(max_triplets, centroids.shape[0] * 3)
    indices = torch.randint(0, centroids.shape[0], (triplet_count, 3), device=features.device)
    a = centroids[indices[:, 0]]
    b = centroids[indices[:, 1]]
    c = centroids[indices[:, 2]]
    ab = a - b
    ac = a - c
    areas = 0.5 * torch.sqrt((ab.square().sum(1) * ac.square().sum(1) - (ab * ac).sum(1).square()).clamp(min=0.0))
    return areas.mean().item()


def _redundancy(candidate: torch.Tensor, selected: List[torch.Tensor]) -> float:
    values = []
    for existing in selected:
        numerator = torch.norm(candidate.T @ existing)
        denominator = torch.norm(candidate) * torch.norm(existing) + 1e-8
        values.append((numerator / denominator).item())
    return max(values) if values else 0.0


def select_layers_from_custom_embeddings(
    embeddings: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    k: Optional[int] = None,
    *,
    task: str = "classification",
    reg: float = 1e-3,
    alpha: float = 1.0,
    gamma: float = 0.5,
    eta: float = 0.1,
    show_progress: ProgressSetting = "auto",
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> LOESResult:
    """Run LOES on precomputed embeddings of shape (n_cal, L, D)."""

    log = logger or LOGGER
    task = _normalize_task(task)
    if embeddings.ndim != 3:
        raise ValueError("embeddings must have shape (n_cal, L, D)")
    if k is None:
        raise ValueError("k must be provided")
    if k < 1:
        raise ValueError("k must be >= 1")

    n_cal, n_layers, _ = embeddings.shape
    if n_cal < 1:
        raise ValueError("embeddings must contain at least one calibration sample")
    if n_layers < 1:
        raise ValueError("embeddings must contain at least one layer")
    if not torch.is_floating_point(embeddings):
        embeddings = embeddings.float()

    _log_info(
        log,
        verbose,
        "Starting LOES on precomputed embeddings: samples=%d layers=%d dim=%d task=%s k=%d",
        n_cal,
        n_layers,
        embeddings.shape[-1],
        task,
        min(k, n_layers),
    )

    layer_views = [embeddings[:, layer_index, :] for layer_index in range(n_layers)]
    if task == "label_free":
        selected, scores = _loes_select_label_free(
            layer_views=layer_views,
            k=min(k, n_layers),
            alpha=alpha,
            gamma=gamma,
            show_progress=show_progress,
            verbose=verbose,
            logger=log,
        )
    else:
        if targets is None:
            raise ValueError("targets must be provided for classification or regression LOES")
        if not torch.is_tensor(targets):
            targets = torch.as_tensor(targets)
        if targets.shape[0] < n_cal:
            raise ValueError("targets must contain at least as many rows as embeddings calibration samples")

        targets = targets[:n_cal].to(embeddings.device)
        y = _as_float_targets(targets, task=task).to(embeddings.device)
        selected, scores = _loes_select(
            layer_views=layer_views,
            y=y,
            raw_targets=targets,
            task=task,
            k=min(k, n_layers),
            reg=reg,
            alpha=alpha,
            gamma=gamma,
            eta=eta,
            show_progress=show_progress,
            verbose=verbose,
            logger=log,
        )
    _log_info(log, verbose, "Completed LOES selection: selected_layers=%s scores=%s", selected, scores)
    return LOESResult(
        selected_layers=selected,
        layer_scores=scores,
        task=task,
        num_layers_seen=n_layers,
        num_calibration_samples=n_cal,
        pooling="precomputed",
    )


def _loes_select_label_free(
    layer_views: List[torch.Tensor],
    k: int,
    alpha: float,
    gamma: float,
    show_progress: ProgressSetting = "auto",
    verbose: bool = False,
    logger: logging.Logger = LOGGER,
) -> Tuple[List[int], List[float]]:
    isotropy_scores = [
        alpha * (1.0 - compute_isotropy(x))
        for x in _iter_with_progress(
            layer_views,
            show_progress=show_progress,
            desc="LOES: label-free score layers",
            total=len(layer_views),
            unit="layer",
            logger=logger,
            verbose=verbose,
        )
    ]
    if not isotropy_scores:
        raise RuntimeError("LOES could not score any layers.")

    best_index = min(range(len(isotropy_scores)), key=isotropy_scores.__getitem__)
    selected = [best_index]
    selection_scores = [isotropy_scores[best_index]]
    _log_info(
        logger,
        verbose,
        "Initial label-free LOES layer=%d score=%.6f",
        best_index,
        selection_scores[0],
    )

    for _ in _iter_with_progress(
        range(1, k),
        show_progress=show_progress,
        desc="LOES: label-free greedy selection",
        total=max(k - 1, 0),
        unit="layer",
        logger=logger,
        verbose=verbose,
    ):
        candidate_score = float("inf")
        candidate_index = None

        for index, x in enumerate(layer_views):
            if index in selected:
                continue
            score = isotropy_scores[index] + gamma * _redundancy(
                x,
                [layer_views[layer_idx] for layer_idx in selected],
            )
            if score < candidate_score:
                candidate_score = score
                candidate_index = index

        if candidate_index is None:
            break

        selected.append(candidate_index)
        selection_scores.append(candidate_score)
        _log_info(
            logger,
            verbose,
            "Label-free LOES step %d/%d selected layer=%d score=%.6f",
            len(selected),
            k,
            candidate_index,
            candidate_score,
        )

    return selected, selection_scores


def _loes_select(
    layer_views: List[torch.Tensor],
    y: torch.Tensor,
    raw_targets: torch.Tensor,
    task: str,
    k: int,
    reg: float,
    alpha: float,
    gamma: float,
    eta: float,
    show_progress: ProgressSetting = "auto",
    verbose: bool = False,
    logger: logging.Logger = LOGGER,
) -> Tuple[List[int], List[float]]:
    geometry_labels: Optional[torch.Tensor] = None
    if task == "classification" and raw_targets.ndim == 1:
        geometry_labels = raw_targets.long()

    initial_scores: List[float] = []
    best_score = float("inf")
    best_index = -1
    best_head: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    _log_info(logger, verbose, "Scoring %d candidate layers for the initial LOES pass", len(layer_views))
    for index, x in enumerate(
        _iter_with_progress(
            layer_views,
            show_progress=show_progress,
            desc="LOES: score layers",
            total=len(layer_views),
            unit="layer",
            logger=logger,
            verbose=verbose,
        )
    ):
        weights, bias = closed_form_ridge(x, y, reg=reg)
        loss = ((x @ weights + bias - y) ** 2).mean().item()
        score = loss + alpha * (1.0 - compute_isotropy(x))
        initial_scores.append(score)
        if score < best_score:
            best_score = score
            best_index = index
            best_head = (weights, bias)

    if best_head is None:
        raise RuntimeError("LOES could not score any layers.")

    selected = [best_index]
    selection_scores = [best_score]
    x_selected = layer_views[best_index].clone()
    y_hat = layer_views[best_index] @ best_head[0] + best_head[1]
    residual = y - y_hat
    _log_info(logger, verbose, "Initial LOES layer=%d score=%.6f", best_index, best_score)

    selection_steps = range(1, k)
    for _ in _iter_with_progress(
        selection_steps,
        show_progress=show_progress,
        desc="LOES: greedy selection",
        total=max(k - 1, 0),
        unit="layer",
        logger=logger,
        verbose=verbose,
    ):
        candidate_score = float("inf")
        candidate_index = None
        candidate_prediction: Optional[torch.Tensor] = None

        for index, x in enumerate(layer_views):
            if index in selected:
                continue

            x_centered = x - x.mean(dim=0, keepdim=True)
            selected_centered = x_selected - x_selected.mean(dim=0, keepdim=True)
            eye = torch.eye(selected_centered.shape[1], device=x.device, dtype=x.dtype)
            orthogonal_map = torch.linalg.solve(
                selected_centered.T @ selected_centered + 1e-6 * eye,
                selected_centered.T @ x_centered,
            )
            x_tilde = x_centered - selected_centered @ orthogonal_map + x.mean(dim=0, keepdim=True)
            weights, bias = closed_form_ridge(x_tilde, residual, reg=reg)
            residual_loss = ((x_tilde @ weights + bias - residual) ** 2).mean().item()
            score = residual_loss + alpha * (1.0 - compute_isotropy(x)) + gamma * _redundancy(
                x, [layer_views[layer_idx] for layer_idx in selected]
            )
            if geometry_labels is not None:
                score = score - eta * _triangle_score(x_tilde, geometry_labels)
            if score < candidate_score:
                candidate_score = score
                candidate_index = index
                candidate_prediction = x_tilde @ weights + bias

        if candidate_index is None:
            break

        if candidate_prediction is None:
            raise RuntimeError("LOES selected a candidate layer without a fitted residual prediction.")

        y_hat = y_hat + candidate_prediction
        residual = y - y_hat
        x_selected = torch.cat([x_selected, layer_views[candidate_index]], dim=1)
        selected.append(candidate_index)
        selection_scores.append(candidate_score)
        _log_info(
            logger,
            verbose,
            "Greedy LOES step %d/%d selected layer=%d score=%.6f residual_mse=%.6f",
            len(selected),
            k,
            candidate_index,
            candidate_score,
            residual.square().mean().item(),
        )

    return selected, selection_scores


def _resolve_device(device: Optional[Union[str, torch.device]], model: torch.nn.Module) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _infer_model_type(model: torch.nn.Module) -> str:
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    return str(model_type) if model_type is not None else ""


def _select_pooling(pooling: str, model: torch.nn.Module) -> str:
    if pooling != "auto":
        return pooling

    model_type = _infer_model_type(model)
    if model_type in TEXT_CLS_MODEL_TYPES or model_type in IMAGE_CLS_MODEL_TYPES:
        return "cls"
    if model_type in {"clip", "clip_vision_model", "vit_mae"}:
        return "cls"
    if model_type in AUDIO_MEAN_MODEL_TYPES:
        return "mean"
    return "mean"


def _pool_hidden_state(
    hidden_state: torch.Tensor,
    *,
    pooling: str,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if hidden_state.ndim == 2:
        return hidden_state
    if hidden_state.ndim != 3:
        raise ValueError(f"Expected a hidden state with 2 or 3 dims, got shape {tuple(hidden_state.shape)}")

    if pooling == "cls":
        return hidden_state[:, 0, :]
    if pooling == "mean":
        return hidden_state.mean(dim=1)
    if pooling == "masked_mean":
        if attention_mask is None:
            raise ValueError("masked_mean pooling requires an attention_mask")
        mask = attention_mask.to(hidden_state.device).unsqueeze(-1).type_as(hidden_state)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (hidden_state * mask).sum(dim=1) / denom
    raise ValueError("pooling must be one of: auto, cls, mean, masked_mean")


def _split_batch(batch: Batch, *, require_targets: bool = True) -> Tuple[Any, Optional[torch.Tensor]]:
    if isinstance(batch, dict):
        for target_key in TARGET_KEYS:
            if target_key in batch:
                inputs = {key: value for key, value in batch.items() if key != target_key}
                return inputs, batch[target_key]
        if not require_targets:
            return batch, None
        raise ValueError("Dict batches must include one of: labels, label, targets, target, y.")
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        return batch[0], batch[1]
    if not require_targets:
        if isinstance(batch, (tuple, list)) and len(batch) == 1:
            return batch[0], None
        return batch, None
    raise ValueError("Each batch must be either (inputs, targets) or a dict with a target key.")


def _to_model_inputs(inputs: Any, device: torch.device) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    if isinstance(inputs, dict):
        return (), {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    if isinstance(inputs, (tuple, list)):
        args = tuple(value.to(device) if torch.is_tensor(value) else value for value in inputs)
        return args, {}
    if torch.is_tensor(inputs):
        return (inputs.to(device),), {}
    raise ValueError("Unsupported input type for model forward pass.")


def _stack_values(values: Sequence[Any]) -> torch.Tensor:
    if not values:
        raise ValueError("Cannot stack an empty batch.")
    first = values[0]
    if torch.is_tensor(first):
        try:
            return torch.stack([value if torch.is_tensor(value) else torch.as_tensor(value) for value in values])
        except RuntimeError:
            return torch.as_tensor(values)
    return torch.as_tensor(values)


def _infer_row_key(row: Dict[str, Any], explicit: Optional[str], candidates: Sequence[str], *, required: bool) -> Optional[str]:
    if explicit is not None:
        if explicit not in row:
            raise ValueError(f"Requested key '{explicit}' was not found in dataset row keys: {sorted(row.keys())}")
        return explicit
    for key in candidates:
        if key in row:
            return key
    if required:
        raise ValueError(f"Could not infer a required key from candidates: {', '.join(candidates)}")
    return None


def _load_hf_model(model_id: str, *, trust_remote_code: bool) -> torch.nn.Module:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError(
            "select_layers_from_hf_id requires transformers. Install with `pip install -e '.[huggingface]'`."
        ) from exc
    return AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code)


def _load_hf_tokenizer(model_id: str, *, trust_remote_code: bool) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Tokenizing raw text datasets requires transformers. Install with `pip install -e '.[huggingface]'`."
        ) from exc
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)


def _load_hf_dataset(dataset_id: str, *, dataset_config_name: Optional[str], split: str) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Loading a dataset by Hugging Face id requires datasets. Install with `pip install -e '.[huggingface]'`."
        ) from exc
    if dataset_config_name is None:
        return load_dataset(dataset_id, split=split)
    return load_dataset(dataset_id, dataset_config_name, split=split)


def _is_torch_dataloader(value: Any) -> bool:
    try:
        from torch.utils.data import DataLoader
    except ImportError:
        return False
    return isinstance(value, DataLoader)


def _build_hf_dataset_dataloader(
    dataset: Any,
    *,
    model_id: str,
    task: str,
    batch_size: int,
    text_key: Optional[str],
    target_key: Optional[str],
    tokenizer: Optional[Any],
    collate_fn: Optional[Callable[[List[Any]], Batch]],
    max_length: int,
    trust_remote_code: bool,
    num_workers: int,
) -> Iterable[Batch]:
    from torch.utils.data import DataLoader

    if collate_fn is not None:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    if len(dataset) == 0:
        raise ValueError("dataset must contain at least one row")

    first_row = dataset[0]
    if not isinstance(first_row, dict):
        raise ValueError("Automatic Hugging Face dataset collation expects dict-like rows.")

    require_targets = task != "label_free"
    inferred_target_key = _infer_row_key(first_row, target_key, TARGET_KEYS, required=require_targets)
    model_input_keys = [key for key in MODEL_INPUT_KEYS if key in first_row]

    if model_input_keys:
        def collate_model_inputs(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
            batch = {key: _stack_values([row[key] for row in rows]) for key in model_input_keys}
            if inferred_target_key is not None:
                batch["labels"] = _stack_values([row[inferred_target_key] for row in rows])
            return batch

        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_model_inputs, num_workers=num_workers)

    inferred_text_key = _infer_row_key(first_row, text_key, TEXT_KEYS, required=True)
    text_tokenizer = tokenizer or _load_hf_tokenizer(model_id, trust_remote_code=trust_remote_code)

    def collate_text(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        encoded = text_tokenizer(
            [str(row[inferred_text_key]) for row in rows],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if inferred_target_key is not None:
            encoded["labels"] = _stack_values([row[inferred_target_key] for row in rows])
        return dict(encoded)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_text, num_workers=num_workers)


def _collect_hf_embeddings(
    model: torch.nn.Module,
    dataloader: Iterable[Batch],
    *,
    max_calibration_samples: Optional[int],
    device: torch.device,
    pooling: str,
    require_targets: bool = True,
    show_progress: ProgressSetting = "auto",
    verbose: bool = False,
    logger: logging.Logger = LOGGER,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    model.eval()
    effective_pooling = _select_pooling(pooling, model)
    layer_storage: Optional[List[List[torch.Tensor]]] = None
    target_storage: List[torch.Tensor] = []
    collected_samples = 0

    _log_info(
        logger,
        verbose,
        "Collecting Hugging Face hidden states: model=%s device=%s pooling=%s max_calibration_samples=%s",
        getattr(getattr(model, "config", None), "_name_or_path", None) or model.__class__.__name__,
        device,
        effective_pooling,
        max_calibration_samples,
    )

    with torch.no_grad():
        for batch in _iter_with_progress(
            dataloader,
            show_progress=show_progress,
            desc="LOES: collect embeddings",
            total=_safe_len(dataloader),
            unit="batch",
            logger=logger,
            verbose=verbose,
        ):
            inputs, targets = _split_batch(batch, require_targets=require_targets)
            args, kwargs = _to_model_inputs(inputs, device=device)
            kwargs["output_hidden_states"] = True
            outputs = model(*args, **kwargs)
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None:
                raise ValueError("The provided model did not return hidden_states. Use a Hugging Face model or compatible wrapper.")

            attention_mask = kwargs.get("attention_mask")
            pooled_layers = [
                _pool_hidden_state(hidden_state, pooling=effective_pooling, attention_mask=attention_mask).detach().cpu()
                for hidden_state in hidden_states[1:]
            ]
            if not pooled_layers:
                raise ValueError("The provided model returned hidden_states but no layer outputs after hidden_states[1:].")

            if layer_storage is None:
                layer_storage = [[] for _ in pooled_layers]
                _log_info(
                    logger,
                    verbose,
                    "Detected %d hidden-state layers with pooled dimension=%d",
                    len(pooled_layers),
                    pooled_layers[0].shape[-1],
                )
            for index, pooled in enumerate(pooled_layers):
                layer_storage[index].append(pooled)

            if targets is not None:
                if not torch.is_tensor(targets):
                    targets = torch.as_tensor(targets)
                targets = targets.detach().cpu()
                if targets.ndim == 0:
                    targets = targets.unsqueeze(0)
                target_storage.append(targets)
                collected_samples += targets.shape[0]
            else:
                collected_samples += pooled_layers[0].shape[0]

            if max_calibration_samples is not None and collected_samples >= max_calibration_samples:
                _log_info(
                    logger,
                    verbose,
                    "Reached calibration sample limit: collected=%d limit=%d",
                    collected_samples,
                    max_calibration_samples,
                )
                break

    if layer_storage is None:
        raise ValueError("No calibration samples were collected from the dataloader.")

    stacked_targets = torch.cat(target_storage, dim=0) if target_storage else None
    if max_calibration_samples is not None and stacked_targets is not None:
        stacked_targets = stacked_targets[:max_calibration_samples]

    stacked_layers = []
    for per_layer in layer_storage:
        layer_tensor = torch.cat(per_layer, dim=0)
        if max_calibration_samples is not None:
            layer_tensor = layer_tensor[:max_calibration_samples]
        stacked_layers.append(layer_tensor)

    embeddings = torch.stack(stacked_layers, dim=1).to(device)
    _log_info(
        logger,
        verbose,
        "Collected calibration tensor: embeddings_shape=%s targets_shape=%s",
        tuple(embeddings.shape),
        tuple(stacked_targets.shape) if stacked_targets is not None else None,
    )
    return embeddings, stacked_targets.to(device) if stacked_targets is not None else None


def select_layers_from_hf(
    model: torch.nn.Module,
    dataset: Optional[Any],
    dataloader: Iterable[Batch],
    *,
    task: str,
    k: int,
    max_calibration_samples: Optional[int] = None,
    pooling: str = "auto",
    device: Optional[Union[str, torch.device]] = None,
    reg: float = 1e-3,
    alpha: float = 1.0,
    gamma: float = 0.5,
    eta: float = 0.1,
    show_progress: ProgressSetting = "auto",
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> LOESResult:
    """
    Run LOES directly on a Hugging Face model and a PyTorch dataloader.

    The dataloader can yield:
    - `(inputs, targets)`
    - `{"input_ids": ..., "attention_mask": ..., "labels": ...}`
    - `{"pixel_values": ..., "labels": ...}`
    - `{"input_values": ..., "labels": ...}`
    """

    log = logger or LOGGER
    task = _normalize_task(task)
    resolved_device = _resolve_device(device, model)
    model = model.to(resolved_device)
    _log_info(log, verbose, "Starting Hugging Face LOES run on device=%s task=%s k=%d", resolved_device, task, k)
    embeddings, targets = _collect_hf_embeddings(
        model,
        dataloader,
        max_calibration_samples=max_calibration_samples,
        device=resolved_device,
        pooling=pooling,
        require_targets=task != "label_free",
        show_progress=show_progress,
        verbose=verbose,
        logger=log,
    )
    result = select_layers_from_custom_embeddings(
        embeddings,
        targets,
        k,
        task=task,
        reg=reg,
        alpha=alpha,
        gamma=gamma,
        eta=eta,
        show_progress=show_progress,
        verbose=verbose,
        logger=log,
    )
    result.dataset = dataset
    result.model_name = getattr(getattr(model, "config", None), "_name_or_path", None) or model.__class__.__name__
    result.pooling = _select_pooling(pooling, model)
    _log_info(log, verbose, "Finished Hugging Face LOES run: selected_layers=%s", result.selected_layers)
    return result


def select_layers_from_hf_id(
    model_id: str,
    dataset: Optional[Any] = None,
    *,
    dataloader: Optional[Iterable[Batch]] = None,
    task: str = "classification",
    k: int,
    dataset_config_name: Optional[str] = None,
    split: str = "train",
    batch_size: int = 16,
    text_key: Optional[str] = None,
    target_key: Optional[str] = None,
    tokenizer: Optional[Any] = None,
    collate_fn: Optional[Callable[[List[Any]], Batch]] = None,
    model_loader: Optional[ModelLoader] = None,
    trust_remote_code: bool = False,
    max_length: int = 512,
    num_workers: int = 0,
    max_calibration_samples: Optional[int] = None,
    pooling: str = "auto",
    device: Optional[Union[str, torch.device]] = None,
    reg: float = 1e-3,
    alpha: float = 1.0,
    gamma: float = 0.5,
    eta: float = 0.1,
    show_progress: ProgressSetting = "auto",
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> LOESResult:
    """
    Load a Hugging Face model by id and run LOES on either a PyTorch dataloader
    or a Hugging Face dataset object/id.
    """

    log = logger or LOGGER
    task = _normalize_task(task)
    model = model_loader(model_id) if model_loader is not None else _load_hf_model(model_id, trust_remote_code=trust_remote_code)

    dataset_metadata = dataset
    effective_dataloader = dataloader
    if effective_dataloader is None:
        if dataset is None:
            raise ValueError("Provide either dataloader=... or dataset=... for select_layers_from_hf_id.")
        if _is_torch_dataloader(dataset):
            effective_dataloader = dataset
            dataset_metadata = "pytorch_dataloader"
        else:
            hf_dataset = _load_hf_dataset(dataset, dataset_config_name=dataset_config_name, split=split) if isinstance(dataset, str) else dataset
            effective_dataloader = _build_hf_dataset_dataloader(
                hf_dataset,
                model_id=model_id,
                task=task,
                batch_size=batch_size,
                text_key=text_key,
                target_key=target_key,
                tokenizer=tokenizer,
                collate_fn=collate_fn,
                max_length=max_length,
                trust_remote_code=trust_remote_code,
                num_workers=num_workers,
            )
            dataset_metadata = dataset if isinstance(dataset, str) else hf_dataset

    _log_info(log, verbose, "Running LOES from Hugging Face id: model_id=%s task=%s", model_id, task)
    result = select_layers_from_hf(
        model=model,
        dataset=dataset_metadata,
        dataloader=effective_dataloader,
        task=task,
        k=k,
        max_calibration_samples=max_calibration_samples,
        pooling=pooling,
        device=device,
        reg=reg,
        alpha=alpha,
        gamma=gamma,
        eta=eta,
        show_progress=show_progress,
        verbose=verbose,
        logger=log,
    )
    result.model_name = model_id
    return result
