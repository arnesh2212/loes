import random
import os
import math
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig, get_cosine_schedule_with_warmup
import tqdm
import csv
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset, load_from_disk

# Optional imports for advanced metrics
try:
    import repitl.matrix_itl as itl
    import repitl.difference_of_entropies as dent
    HAS_REPITL = True
except ImportError:
    HAS_REPITL = False
    print("Warning: repitl not installed. Some metrics (entropy, LIDAR, DIME) will be unavailable.")

try:
    from dadapy.data import Data as ID_DATA
    HAS_DADAPY = True
except ImportError:
    HAS_DADAPY = False
    print("Warning: dadapy not installed. Intrinsic dimension metric will be unavailable.")


class BertEncoder(nn.Module):
    def __init__(self, model_name_or_path="bert-base-uncased"):
        super().__init__()
        if os.path.isdir(model_name_or_path):
            config = AutoConfig.from_pretrained(model_name_or_path)
            self.backbone = AutoModel.from_pretrained(model_name_or_path, config=config)
        else:
            self.backbone = AutoModel.from_pretrained(model_name_or_path)
        self.hidden_size = self.backbone.config.hidden_size

    def forward(self, input_ids, attention_mask, return_layers=False):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        if return_layers:
            return [h[:, 0, :] for h in out.hidden_states[1:]]
        return out.last_hidden_state[:, 0, :]


class TextDataset:
    def __init__(self, dataset_path_or_name, subset, split, tokenizer, max_len=128):
        offline_subset_path = os.path.join(dataset_path_or_name, f"{subset}_{split}") if subset else os.path.join(dataset_path_or_name, split)
        
        if os.path.exists(offline_subset_path):
            self.ds = load_from_disk(offline_subset_path)
        else:
            if subset:
                self.ds = load_dataset(dataset_path_or_name, subset, split=split)
            else:
                self.ds = load_dataset(dataset_path_or_name, split=split)
            save_path = f"./offline_data/{subset if subset else 'full'}_{split}"
            self.ds.save_to_disk(save_path)

        self.tokenizer = tokenizer
        self.max_len = max_len
        
        self.text_key = next((k for k in self.ds.features if k in ['text', 'sentence', 'sentence1', 'content']), None)
        self.label_key = next((k for k in self.ds.features if k in ['label', 'labels', 'coarse_label', 'fine_label']), None)

        if not self.text_key or not self.label_key:
            raise ValueError(f"Columns not found. Available: {self.ds.column_names}")

        label_feature = self.ds.features[self.label_key]

        if hasattr(label_feature, "names"):
            self.label2id = {name: i for i, name in enumerate(label_feature.names)}
        else:
            sample = self.ds[0][self.label_key]
            if isinstance(sample, str):
                uniq = sorted(set(self.ds[self.label_key]))
                self.label2id = {v: i for i, v in enumerate(uniq)}
                self.ds = self.ds.map(lambda x: {self.label_key: self.label2id[x[self.label_key]]})
            else:
                self.label2id = None

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        return item[self.text_key], item[self.label_key]


# ==================== UTILITY FUNCTIONS ====================

def entropy_normalization(entropy, normalization, N, D):
    """Normalize entropy based on specified method."""
    assert normalization in ['maxEntropy', 'logN', 'logD', 'logNlogD', 'raw', 'length']
    
    if normalization == 'maxEntropy':
        entropy /= min(math.log(N), math.log(D))
    elif normalization == 'logN':
        entropy /= math.log(N)
    elif normalization == 'logD':
        entropy /= math.log(D)
    elif normalization == 'logNlogD':
        entropy /= (math.log(N) * math.log(D))
    elif normalization == 'raw':
        pass
    elif normalization == 'length':
        entropy = N
    return entropy


def compute_isotropy(X, eps=1e-9):
    Xc = X - X.mean(0, keepdim=True)
    eigs = torch.linalg.eigvalsh((Xc.t() @ Xc) / Xc.shape[0]).real.clamp(min=0.0)
    return (eigs.mean() / (eigs.std(unbiased=False) + eps)).item()


def closed_form_ridge(X, Y, reg=1e-3):
    Xc, Yc = X - X.mean(0, keepdim=True), Y - Y.mean(0, keepdim=True)
    W = torch.linalg.solve(Xc.t() @ Xc + reg * torch.eye(X.shape[1], device=X.device), Xc.t() @ Yc)
    b = (Y.mean(0, keepdim=True) - X.mean(0, keepdim=True) @ W).squeeze(0)
    return W, b


def collect_calibration_embeddings(net, loader, n_cal, device="cuda"):
    net.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids, mask, y = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            f = [t.cpu() for t in net(ids, mask, return_layers=True)]
            if not embeddings: embeddings = [[] for _ in f]
            for i, t in enumerate(f): embeddings[i].append(t)
            labels.append(y.cpu())
            if sum(len(b) for b in labels) >= n_cal: break
    return [torch.cat(e)[:n_cal].to(device) for e in embeddings], torch.cat(labels)[:n_cal].long().to(device)


# ==================== METRIC COMPUTATION FUNCTIONS ====================

def compute_entropy_metric(hidden_states, alpha=1, normalization='maxEntropy'):
    """Compute matrix entropy for each layer."""
    if not HAS_REPITL:
        raise ImportError("repitl required for entropy metric")
    
    L = len(hidden_states)
    entropies = []
    
    for layer_emb in hidden_states:
        X = layer_emb.double()
        N, D = X.shape
        
        if N > D:
            cov = X.T @ X
        else:
            cov = X @ X.T
        
        cov = torch.clamp(cov, min=0)
        try:
            cov_norm = cov / torch.trace(cov)
            ent = itl.matrixAlphaEntropy(cov_norm, alpha=alpha).item()
            entropies.append(entropy_normalization(ent, normalization, N, D))
        except Exception:
            entropies.append(float('nan'))
    
    return entropies


def compute_curvature_metric(hidden_states):
    """Compute average curvature for each layer."""
    L = len(hidden_states)
    curvatures = []
    
    for layer_emb in hidden_states:
        X = layer_emb.double()
        N, D = X.shape
        
        if N < 3:
            curvatures.append(0.0)
            continue
        
        total_curv = 0.0
        count = 0
        
        for k in range(1, N - 1):
            v_k = (X[k] - X[k-1]).unsqueeze(1)
            v_kplus1 = (X[k+1] - X[k]).unsqueeze(1)
            
            dot = torch.abs(v_k.T @ v_kplus1)
            norm_a, norm_b = torch.norm(v_k), torch.norm(v_kplus1)
            
            if norm_a > 0 and norm_b > 0:
                arg = torch.clamp(dot / (norm_a * norm_b), min=-1, max=1)
                curv = torch.arccos(arg).item()
                if not math.isnan(curv):
                    total_curv += curv
                    count += 1
        
        curvatures.append(total_curv / count if count > 0 else 0.0)
    
    return curvatures


def compute_intrinsic_dimension_metric(hidden_states):
    """Compute intrinsic dimension using TwoNN estimator."""
    if not HAS_DADAPY:
        raise ImportError("dadapy required for intrinsic dimension metric")
    
    intrinsic_dims = []
    
    for layer_emb in hidden_states:
        X = layer_emb.detach().float().cpu().numpy()
        try:
            data = ID_DATA(X)
            id_val, _, _ = data.compute_id_2NN()
            intrinsic_dims.append(id_val)
        except Exception:
            intrinsic_dims.append(float('nan'))
    
    return intrinsic_dims


def compute_linear_probe_loss(hidden_states, labels, reg=1e-3):
    """Compute ridge regression loss for each layer."""
    Y = F.one_hot(labels, int(labels.max()) + 1).float()
    losses = []
    
    for X in hidden_states:
        W, b = closed_form_ridge(X, Y, reg)
        loss = ((X @ W + b - Y) ** 2).mean().item()
        losses.append(loss)
    
    return losses


def compute_class_separability(hidden_states, labels):
    """Compute class separability (between-class / within-class variance ratio)."""
    separabilities = []
    classes = torch.unique(labels)
    num_classes = len(classes)
    
    for X in hidden_states:
        if num_classes < 2:
            separabilities.append(0.0)
            continue
        
        # Global mean
        global_mean = X.mean(dim=0)
        
        # Between-class scatter
        between = 0.0
        within = 0.0
        
        for c in classes:
            mask = labels == c
            class_samples = X[mask]
            n_c = class_samples.shape[0]
            
            if n_c == 0:
                continue
            
            class_mean = class_samples.mean(dim=0)
            
            # Between-class
            diff = class_mean - global_mean
            between += n_c * (diff @ diff)
            
            # Within-class
            centered = class_samples - class_mean
            within += (centered ** 2).sum()
        
        sep = (between / (within + 1e-8)).item()
        separabilities.append(sep)
    
    return separabilities


def compute_dime_metric(hidden_states, alpha=1.0, normalization='maxEntropy'):
    """
    Computes DIME (Difference of Entropies). 
    Expects hidden_states to have shape [L, N, 2, D] or be a list of [N, 2, D].
    """
    if not HAS_REPITL:
        raise ImportError("repitl required for DIME")
    
    dimes = []
    for layer_emb in hidden_states:
        # layer_emb: [N, 2, D]
        view_a = layer_emb[:, 0, :].double()
        view_b = layer_emb[:, 1, :].double()
        N, D = view_a.shape
        
        try:
            # doe computes H(A) + H(B) - H(A, B) using matrix ITL
            val = dent.doe(view_a, view_b, alpha=alpha, n_iters=10).item()
            dimes.append(entropy_normalization(val, normalization, N, D))
        except:
            dimes.append(float('nan'))
    return dimes

def compute_infonce_metric(hidden_states, temperature=0.1):
    """
    Computes InfoNCE based MI lower bound.
    Expects hidden_states to have shape [L, N, 2, D].
    """
    scores = []
    for layer_emb in hidden_states:
        view_a = F.normalize(layer_emb[:, 0, :].float(), dim=-1)
        view_b = F.normalize(layer_emb[:, 1, :].float(), dim=-1)
        N = view_a.shape[0]
        
        logits = (view_a @ view_b.T) / temperature
        labels = torch.arange(N, device=logits.device)
        loss = F.cross_entropy(logits, labels).item()
        
        # Normalized score: 1 - (loss / log(N))
        score = 1.0 - (loss / math.log(N))
        scores.append(score)
    return scores



def compute_centroid_distances(hidden_states, labels):
    """Compute average pairwise centroid distances."""
    classes = torch.unique(labels)
    distances = []
    
    for X in hidden_states:
        centroids = []
        for c in classes:
            mask = labels == c
            if mask.sum() > 0:
                centroids.append(X[mask].mean(dim=0))
        
        if len(centroids) < 2:
            distances.append(0.0)
            continue
        
        centroids = torch.stack(centroids)
        n = len(centroids)
        total_dist = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += torch.norm(centroids[i] - centroids[j]).item()
                count += 1
        
        distances.append(total_dist / count if count > 0 else 0.0)
    
    return distances


# ==================== LAYER SELECTION METHODS ====================

def loes_select_layers(embeddings, labels, K, reg=1e-3, alpha=1.0, gamma=0.5, eta=0.1):
    """Original LOES layer selection."""
    Y = F.one_hot(labels, int(labels.max())+1).float()
    best = (float("inf"), -1, None)
    for i, X in enumerate(embeddings):
        W, b = closed_form_ridge(X, Y, reg)
        loss = ((X @ W + b - Y)**2).mean().item()
        iso = compute_isotropy(X)
        if (loss + alpha*(1-iso)) < best[0]: best = (loss + alpha*(1-iso), i, (W, b))
    
    selected = [best[1]]
    X_S = embeddings[best[1]].clone()
    y_hat = embeddings[best[1]] @ best[2][0] + best[2][1]
    residual = Y - y_hat
    
    while len(selected) < K:
        best = (float("inf"), None, None)
        for i, X in enumerate(embeddings):
            if i in selected: continue
            Xc, XS_c = X - X.mean(0, keepdim=True), X_S - X_S.mean(0, keepdim=True)
            I = torch.eye(XS_c.shape[1], device=XS_c.device, dtype=XS_c.dtype)
            B_orth = torch.linalg.solve(XS_c.t() @ XS_c + 1e-6 * I, XS_c.t() @ Xc)
            X_tilde = Xc - XS_c @ B_orth + X.mean(0, keepdim=True)
            W, b = closed_form_ridge(X_tilde, residual, reg)
            res_loss = ((X_tilde @ W + b - residual)**2).mean().item()
            iso = compute_isotropy(X)
            red = max([ (torch.norm(X.t()@embeddings[j])/(torch.norm(X)*torch.norm(embeddings[j]))).item() for j in selected])
            classes = torch.unique(labels)
            cents = torch.stack([X_tilde[labels==c].mean(0) for c in classes]) if len(classes)>=3 else None
            tri = 0.0
            if cents is not None:
                idx = torch.randint(0, len(cents), (200, 3))
                a, b_pt, c_pt = cents[idx[:,0]], cents[idx[:,1]], cents[idx[:,2]]
                ab, ac = a-b_pt, a-c_pt
                tri = 0.5 * torch.sqrt((ab.pow(2).sum(1)*ac.pow(2).sum(1)-(ab*ac).sum(1).pow(2)).clamp(min=0)).mean().item()
            score = res_loss + alpha*(1-iso) + gamma*red - eta*tri
            if score < best[0]: best = (score, i, (W, b, X_tilde))
        if best[1] is None: break
        idx, (W, b, _) = best[1], best[2]
        W_f, b_f = closed_form_ridge(embeddings[idx], residual + y_hat, reg)
        y_hat += embeddings[idx] @ W_f + b_f
        residual = Y - y_hat
        X_S = torch.cat([X_S, embeddings[idx]], dim=1)
        selected.append(idx)
    return selected


def select_by_metric(embeddings, labels, K, metric_name, higher_is_better=True):
    """Generic layer selection by a single metric."""
    # Compute the metric for all layers
    if metric_name == "entropy":
        scores = compute_entropy_metric(embeddings)
    elif metric_name == "curvature":
        scores = compute_curvature_metric(embeddings)
    elif metric_name == "intrinsic_dim":
        scores = compute_intrinsic_dimension_metric(embeddings)
    elif metric_name == "probe_loss":
        scores = compute_linear_probe_loss(embeddings, labels)
        higher_is_better = False  # Lower loss is better
    elif metric_name == "isotropy":
        scores = [compute_isotropy(X) for X in embeddings]
    elif metric_name == "separability":
        scores = compute_class_separability(embeddings, labels)
    elif metric_name == "centroid_dist":
        scores = compute_centroid_distances(embeddings, labels)
    elif metric_name == "dime":
        scores = compute_dime_metric(embeddings)
    elif metric_name == "infonce":
        scores = compute_infonce_metric(embeddings)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")
    
    # Handle NaN values
    scores = [s if not math.isnan(s) else (float('-inf') if higher_is_better else float('inf')) for s in scores]
    
    # Sort and select top K
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=higher_is_better)
    
    selected = [idx for idx, _ in indexed_scores[:K]]
    return selected


def select_by_combined_score(embeddings, labels, K, metrics_config):
    """
    Select layers by combining multiple metrics.
    metrics_config: list of (metric_name, weight, higher_is_better)
    """
    L = len(embeddings)
    combined_scores = [0.0] * L
    
    for metric_name, weight, higher_is_better in metrics_config:
        if metric_name == "entropy":
            scores = compute_entropy_metric(embeddings)
        elif metric_name == "curvature":
            scores = compute_curvature_metric(embeddings)
        elif metric_name == "intrinsic_dim":
            scores = compute_intrinsic_dimension_metric(embeddings)
        elif metric_name == "probe_loss":
            scores = compute_linear_probe_loss(embeddings, labels)
        elif metric_name == "isotropy":
            scores = [compute_isotropy(X) for X in embeddings]
        elif metric_name == "separability":
            scores = compute_class_separability(embeddings, labels)
        elif metric_name == "centroid_dist":
            scores = compute_centroid_distances(embeddings, labels)
        elif metric_name == "dime":
            scores = compute_dime_metric(embeddings)
        elif metric_name == "infonce":
            scores = compute_infonce_metric(embeddings)
        else:
            continue
        
        # Normalize scores to [0, 1]
        valid_scores = [s for s in scores if not math.isnan(s)]
        if len(valid_scores) > 0:
            min_s, max_s = min(valid_scores), max(valid_scores)
            range_s = max_s - min_s if max_s > min_s else 1.0
            normalized = [(s - min_s) / range_s if not math.isnan(s) else 0.5 for s in scores]
        else:
            normalized = [0.5] * L
        
        # Flip if lower is better
        if not higher_is_better:
            normalized = [1.0 - n for n in normalized]
        
        # Add weighted contribution
        for i in range(L):
            combined_scores[i] += weight * normalized[i]
    
    # Select top K
    indexed_scores = list(enumerate(combined_scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    
    selected = [idx for idx, _ in indexed_scores[:K]]
    return selected


# ==================== TRAINING COMPONENTS ====================

class GeometricLoss(nn.Module):
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, feats, labels):
        if self.weight <= 0: return torch.tensor(0.0, device=feats.device)
        classes = torch.unique(labels)
        if len(classes) < 3: return torch.tensor(0.0, device=feats.device)
        centroids = torch.stack([feats[labels==c].mean(0) for c in classes])
        if centroids.shape[0] < 3: return torch.tensor(0.0, device=feats.device)
        idx = torch.randperm(len(centroids))[:3]
        ab = centroids[idx[0]] - centroids[idx[1]]
        ac = centroids[idx[0]] - centroids[idx[2]]
        area = 0.5 * torch.sqrt((ab.pow(2).sum() * ac.pow(2).sum() - (ab * ac).sum().pow(2)).clamp(min=1e-6))
        cov = torch.cov(feats.T) + 1e-4 * torch.eye(feats.shape[1], device=feats.device)
        iso_loss = torch.linalg.eigvalsh(cov).real.clamp(min=1e-6).var()
        return self.weight * (iso_loss - torch.log(area + 1e-6))


class LearnableWeighting(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n))
    
    def forward(self, x):
        w = F.softmax(self.w, dim=0)
        return sum(x[i] * w[i] for i in range(len(x)))


def run_eval(loader, net, probe, adapters, aggregator, topk, fusion, device):
    net.eval(); probe.eval()
    corr, tot = 0, 0
    with torch.no_grad():
        for batch in loader:
            ids, mask, y = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            f = net(ids, mask, return_layers=True)
            sel = [f[i] for i in topk]
            if adapters:
                sel = [
                    adapters[i](f.to(next(adapters[i].parameters()).dtype))
                    for i, f in enumerate(sel)
                ]            
            if aggregator: emb = aggregator(sel)
            elif fusion == "concat": emb = torch.cat(sel, dim=-1)
            elif fusion == "mean": emb = torch.stack(sel, dim=0).mean(0)
            elif fusion == "sum": emb = torch.stack(sel, dim=0).sum(0)
            
            corr += (probe(emb).argmax(1) == y).sum().item()
            tot += y.shape[0]
    return corr / tot if tot > 0 else 0.0


# ==================== MAIN ====================

@hydra.main(config_path="conf", config_name="config_text", version_base=None)
def main(cfg: DictConfig):
    device = cfg.device
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    for model_path in cfg.model.names:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        def collate_fn(batch):
            texts, labels = zip(*batch)
            toks = tokenizer(list(texts), padding=True, truncation=True, max_length=cfg.model.max_len, return_tensors="pt")
            toks['labels'] = torch.tensor(labels)
            return toks

        for dcfg in cfg.datasets:
            dataset_path = dcfg.get("path", None) or dcfg.name
            subsets = list(dcfg.subsets) if dcfg.subsets else [None]

            for subset in subsets:
                print(f"==========================================")
                print(f"PROCESSING SUBSET: {subset if subset else 'Full Dataset'}")
                print(f"==========================================")

                train_ds = TextDataset(dataset_path, subset, dcfg.train_split, tokenizer)
                
                try:
                    val_ds = TextDataset(dataset_path, subset, dcfg.val_split, tokenizer)
                    print(f"Loaded Validation Split: {len(val_ds)} samples")
                except Exception as e:
                    print(f"Warning: Validation split loading failed ({e}). Using train as validation.")
                    val_ds = train_ds 
                
                if dcfg.test_split:
                    try:
                        test_ds = TextDataset(dataset_path, subset, dcfg.test_split, tokenizer)
                        print(f"Loaded Test Split: {len(test_ds)} samples")
                    except Exception as e:
                        print(f"Test split loading failed ({e}). Using Validation split as Test.")
                        test_ds = val_ds
                else:
                    print("Test split is null in config. Using Validation split as Test.")
                    test_ds = val_ds

                train_loader = DataLoader(train_ds, cfg.training.bs, shuffle=True, collate_fn=collate_fn, num_workers=4)
                cal_loader = DataLoader(train_ds, cfg.calibration.cal_bs, shuffle=True, collate_fn=collate_fn)
                val_loader = DataLoader(val_ds, cfg.training.test_bs, collate_fn=collate_fn, num_workers=4)
                test_loader = DataLoader(test_ds, cfg.training.test_bs, collate_fn=collate_fn, num_workers=4)
                
                experiments = cfg.exps if cfg.exps else ["default"]
                
                for exp_str in experiments:
                    print(f"--- Starting Experiment: {exp_str} on {subset} ---")
                    exp_cfg = cfg.copy()
                    
                    # --- EXPERIMENT PARSING LOGIC ---
                    selection_mode = "last"
                    topk_val = 1
                    use_geo_loss = False
                    fusion_mode = "concat"
                    metric_name = None
                    
                    if exp_str == "last":
                        selection_mode = "last"
                        topk_val = 1
                    elif exp_str == "last_3_mean":
                        selection_mode = "last_k"
                        topk_val = 3
                        fusion_mode = "mean"
                    elif exp_str == "last_3_concat":
                        selection_mode = "last_k"
                        topk_val = 3
                    elif "learnable_weight" in exp_str:
                        selection_mode = "learnable_weight"
                        topk_val = 0 
                        fusion_mode = "sum"
                        use_geo_loss = True
                    elif "loes" in exp_str:
                        selection_mode = "loes"
                        if "_k_" in exp_str: topk_val = int(exp_str.split("_k_")[1].split("_")[0])
                        else: topk_val = cfg.topk
                        if "no_geo" in exp_str: use_geo_loss = False
                        if "mean_fusion" in exp_str: fusion_mode = "mean"
                    elif "random" in exp_str:
                        selection_mode = "random"
                        if "_k_" in exp_str: topk_val = int(exp_str.split("_k_")[1].split("_")[0])
                        else: topk_val = cfg.topk
                    # --- NEW METRIC-BASED SELECTIONS ---
                    elif "entropy_k_" in exp_str:
                        selection_mode = "metric"
                        metric_name = "entropy"
                        topk_val = int(exp_str.split("_k_")[1].split("_")[0])
                    elif "curvature_k_" in exp_str:
                        selection_mode = "metric"
                        metric_name = "curvature"
                        topk_val = int(exp_str.split("_k_")[1].split("_")[0])
                    elif "intrinsic_dim_k_" in exp_str:
                        selection_mode = "metric"
                        metric_name = "intrinsic_dim"
                        topk_val = int(exp_str.split("_k_")[1].split("_")[0])
                    elif "probe_loss_k_" in exp_str:
                        selection_mode = "metric"
                        metric_name = "probe_loss"
                        topk_val = int(exp_str.split("_k_")[1].split("_")[0])
                    elif "isotropy_k_" in exp_str:
                        selection_mode = "metric"
                        metric_name = "isotropy"
                        topk_val = int(exp_str.split("_k_")[1].split("_")[0])
                    elif "separability_k_" in exp_str:
                        selection_mode = "metric"
                        metric_name = "separability"
                        topk_val = int(exp_str.split("_k_")[1].split("_")[0])
                    elif "centroid_dist_k_" in exp_str:
                        selection_mode = "metric"
                        metric_name = "centroid_dist"
                        topk_val = int(exp_str.split("_k_")[1].split("_")[0])
                    elif "combined_k_" in exp_str:
                        selection_mode = "combined"
                        topk_val = int(exp_str.split("_k_")[1].split("_")[0])
                    elif "dime_k_" in exp_str:
                        selection_mode = "metric"
                        metric_name = "dime"
                        topk_val = int(exp_str.split("_k_")[1].split("_")[0])
                    elif "infonce_k_" in exp_str:
                        selection_mode = "metric"
                        metric_name = "infonce"
                        topk_val = int(exp_str.split("_k_")[1].split("_")[0])
                    else:
                        # Default fallback
                        selection_mode = "last"
                        topk_val = 1

                    # Update config values
                    exp_cfg.selection.mode = selection_mode
                    exp_cfg.topk = topk_val
                    exp_cfg.ablation.use_geo_loss = use_geo_loss
                    exp_cfg.ablation.fusion = fusion_mode

                    net = BertEncoder(model_path).to(device)
                    if not exp_cfg.optim.finetune:
                        for p in net.parameters(): p.requires_grad = False
                    n_cal = 0.2 * len(train_ds)
                    embeddings, labels = collect_calibration_embeddings(net, cal_loader, n_cal, device)
                    total_layers = len(embeddings)
                    embeddings = [e.to(torch.float32) for e in embeddings]

                    # --- SELECTION LOGIC ---
                    if selection_mode == "loes": 
                        topk = loes_select_layers(embeddings, labels, K=topk_val)
                    elif selection_mode == "random": 
                        topk = random.sample(range(total_layers), min(topk_val, total_layers))
                    elif selection_mode == "last": 
                        topk = [total_layers-1]
                    elif selection_mode == "last_k":
                        topk = list(range(max(0, total_layers - topk_val), total_layers))
                    elif selection_mode == "learnable_weight":
                        topk = list(range(total_layers))
                    elif selection_mode == "metric":
                        # Metric-based selection
                        higher_is_better = metric_name not in ["probe_loss", "curvature"]  # Lower is better for these
                        try:
                            topk = select_by_metric(embeddings, labels, topk_val, metric_name, higher_is_better)
                        except ImportError as e:
                            print(f"Skipping {exp_str}: {e}")
                            continue
                    elif selection_mode == "combined":
                        # Combined metric selection
                        metrics_config = [
                            ("separability", 1.0, True),
                            ("isotropy", 0.5, True),
                            ("probe_loss", 1.0, False),
                        ]
                        topk = select_by_combined_score(embeddings, labels, topk_val, metrics_config)
                    else: 
                        topk = list(range(total_layers))

                    print(f"Selected Layers: {topk}")
                    D = net.hidden_size
                    proj_dim = exp_cfg.model.proj_dim
                    
                    adapters = None
                    if not exp_cfg.ablation.no_adapters:
                        adapters = nn.ModuleList([nn.Sequential(nn.LayerNorm(D), nn.Linear(D, proj_dim), nn.GELU()) for _ in topk]).to(device)
                    
                    aggregator = None
                    if selection_mode == "learnable_weight":
                        aggregator = LearnableWeighting(len(topk)).to(device)
                        fused_dim = proj_dim
                    elif fusion_mode == "concat": 
                        fused_dim = proj_dim * len(topk)
                    else: 
                        fused_dim = proj_dim

                    num_classes = len(set(train_ds.ds[train_ds.label_key]))
                    probe = nn.Sequential(nn.LayerNorm(fused_dim), nn.Dropout(0.2), nn.Linear(fused_dim, num_classes)).to(device)
                    
                    # --- OPTIMIZER & SCHEDULER ---
                    params = [{"params": probe.parameters(), "lr": exp_cfg.optim.lr_probe}]
                    if adapters: params.append({"params": adapters.parameters(), "lr": exp_cfg.optim.lr_probe})
                    if aggregator: params.append({"params": aggregator.parameters(), "lr": 1e-3})
                    
                    if exp_cfg.optim.finetune: 
                        params.append({"params": net.parameters(), "lr": exp_cfg.optim.lr_backbone})
                    
                    opt = torch.optim.AdamW(params, weight_decay=exp_cfg.optim.weight_decay)
                    
                    total_steps = len(train_loader) * exp_cfg.training.epochs
                    warmup_steps = int(total_steps * exp_cfg.training.warmup_ratio)
                    
                    sched = get_cosine_schedule_with_warmup(
                        opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps
                    )

                    geo_loss = GeometricLoss(weight=0.1 if exp_cfg.ablation.use_geo_loss else 0.0)

                    ds_name = f"{dcfg.name}/{subset}" if subset else dcfg.name
                    run_name = f"{exp_str}_{ds_name.replace('/', '_')}"
                    
                    # --- BEST MODEL SAVING ---
                    best_val_acc = 0.0
                    ckpt_path = f"best_model_{run_name}.pt"
                    
                    wandb.init(project=exp_cfg.wandb.project, name=run_name, config=OmegaConf.to_container(exp_cfg), reinit=True)
                    
                    # Log layer selection info
                    wandb.log({
                        "selection_mode": selection_mode,
                        "metric_name": metric_name if metric_name else "N/A",
                        "selected_layers": str(topk),
                        "num_selected": len(topk)
                    })
                    
                    for ep in range(exp_cfg.training.epochs):
                        net.train() if exp_cfg.optim.finetune else net.eval()
                        probe.train()
                        for batch in tqdm.tqdm(train_loader, desc=f"Ep {ep} - {subset}"):
                            opt.zero_grad()
                            ids, mask, y = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
                            feats = net(ids, mask, return_layers=True)
                            sel_feats = [feats[i] for i in topk]
                            if adapters:
                                sel_feats = [
                                    adapters[i](f.to(next(adapters[i].parameters()).dtype))
                                    for i, f in enumerate(sel_feats)
                                ]
                            
                            if aggregator: emb = aggregator(sel_feats)
                            elif fusion_mode == "concat": emb = torch.cat(sel_feats, dim=-1)
                            elif fusion_mode == "mean": emb = torch.stack(sel_feats, dim=0).mean(0)
                            elif fusion_mode == "sum": emb = torch.stack(sel_feats, dim=0).sum(0)
                            
                            loss = F.cross_entropy(probe(emb), y) + geo_loss(emb, y)
                            loss.backward()
                            
                            torch.nn.utils.clip_grad_norm_(
                                net.parameters() if exp_cfg.optim.finetune else probe.parameters(), 
                                max_norm=1.0
                            )
                            
                            opt.step()
                            sched.step()
                            wandb.log({"train_loss": loss.item()})
                        
                        # Validation
                        val_acc = run_eval(val_loader, net, probe, adapters, aggregator, topk, fusion_mode, device)
                        wandb.log({"val_acc": val_acc})
                        print(f"Ep {ep} Val Acc: {val_acc:.4f}")
                        
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            print(f"New best Val Acc: {val_acc:.4f}. Saving checkpoint to {ckpt_path}")
                            state = {
                                'probe': probe.state_dict(),
                                'adapters': adapters.state_dict() if adapters else None,
                                'aggregator': aggregator.state_dict() if aggregator else None,
                                'net': net.state_dict() if exp_cfg.optim.finetune else None
                            }
                            torch.save(state, ckpt_path)

                    # Test
                    print(f"Loading best model from {ckpt_path} for final Testing...")
                    if os.path.exists(ckpt_path):
                        checkpoint = torch.load(ckpt_path)
                        probe.load_state_dict(checkpoint['probe'])
                        if adapters and checkpoint['adapters']: adapters.load_state_dict(checkpoint['adapters'])
                        if aggregator and checkpoint['aggregator']: aggregator.load_state_dict(checkpoint['aggregator'])
                        if exp_cfg.optim.finetune and checkpoint['net']: net.load_state_dict(checkpoint['net'])
                    else:
                        print("Warning: No checkpoint found! Testing with last epoch weights.")

                    test_acc = run_eval(test_loader, net, probe, adapters, aggregator, topk, fusion_mode, device)
                    wandb.log({"test_acc": test_acc})
                    print(f"Final Test Acc (Best Model): {test_acc:.4f}")

                    with open(exp_cfg.logging.results_csv, "a+", newline='') as f:
                        f.seek(0, os.SEEK_END)
                        w = csv.DictWriter(f, fieldnames=[
                            "run", "mode", "metric", "k", "fusion", "geo", "test_acc", 
                            "dataset", "selected_layers", "subset", "model"
                        ], extrasaction='ignore')
                        if f.tell() == 0: w.writeheader()
                        w.writerow({
                            "run": run_name, 
                            "mode": selection_mode, 
                            "metric": metric_name if metric_name else "N/A",
                            "k": len(topk),
                            "fusion": fusion_mode, 
                            "geo": exp_cfg.ablation.use_geo_loss,
                            "test_acc": test_acc, 
                            "dataset": ds_name, 
                            "selected_layers": topk, 
                            "subset": subset, 
                            "model": model_path
                        })
                    
                    wandb.finish()


if __name__ == "__main__": 
    main()