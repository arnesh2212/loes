import random
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2
import wandb
import hydra
import tqdm
from omegaconf import DictConfig
from datasets import load_dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection, AutoProcessor
from datasets import load_from_disk
from pathlib import Path


LOCAL_DS_DIR_SIDA = Path("/home/arush/deepfake/sida_net/datasets/SID_Set")


# HuggingFace ViT Encoder - Unified interface for all models
# HuggingFace ViT Encoder - Unified interface for all models
# HuggingFace ViT Encoder - Unified interface for all models
class HFViTEncoder(nn.Module):
    # Pooling strategies:
    # - "cls": Use CLS token at index 0 (DINOv2, DINOv3, ViT-IN21k, DeiT)
    # - "mean": Mean pool all tokens (CLIP, MAE)
    
    POOLING_MAP = {
        "openai/clip-vit-base-patch32": "mean",
        "facebook/dinov3-vits16-pretrain-lvd1689m": "cls",
        "facebook/dinov2-small": "cls",
        "facebook/vit-mae-base": "mean",
        "facebook/deit-base-distilled-patch16-224": "cls",
        "google/vit-base-patch16-224-in21k": "cls",
    }
    
    def __init__(self, model_name, img_size=224):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        
        # Determine pooling strategy
        self.pooling = self.POOLING_MAP.get(model_name, "cls")
        print(f"  Loading {model_name} with pooling={self.pooling}")
        
        # Load model based on type - each model family has different structure
        if "clip" in model_name.lower():
            # CLIP: uses CLIPVisionModelWithProjection
            self.model = CLIPVisionModelWithProjection.from_pretrained(model_name)
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.embed_dim = self.model.config.hidden_size
            self.n_layers = self.model.config.num_hidden_layers
            print(f"  CLIP model: embed_dim={self.embed_dim}, n_layers={self.n_layers}")
            
        elif "dinov2" in model_name.lower() and "dinov3" not in model_name.lower():
            # DINOv2: uses Dinov2Model
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.embed_dim = self.model.config.hidden_size
            self.n_layers = self.model.config.num_hidden_layers
            print(f"  DINOv2 model: embed_dim={self.embed_dim}, n_layers={self.n_layers}")
            
        elif "dinov3" in model_name.lower():
            # DINOv3: similar to DINOv2 but may have register tokens
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.embed_dim = self.model.config.hidden_size
            self.n_layers = self.model.config.num_hidden_layers
            print(f"  DINOv3 model: embed_dim={self.embed_dim}, n_layers={self.n_layers}")
            if hasattr(self.model.config, 'num_register_tokens'):
                print(f"  DINOv3 register tokens: {self.model.config.num_register_tokens}")
            
        elif "mae" in model_name.lower():
            # MAE: uses ViTMAEModel
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.embed_dim = self.model.config.hidden_size
            self.n_layers = self.model.config.num_hidden_layers
            print(f"  MAE model: embed_dim={self.embed_dim}, n_layers={self.n_layers}")
            
        elif "deit" in model_name.lower():
            # DeiT: distilled ViT - has CLS + distillation token + patches
            # Use CLS token for pooling (index 0)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.embed_dim = self.model.config.hidden_size
            self.n_layers = self.model.config.num_hidden_layers
            print(f"  DeiT model: embed_dim={self.embed_dim}, n_layers={self.n_layers}")
            
        elif "google/vit" in model_name.lower() or "vit-base" in model_name.lower():
            # Google ViT: standard ViT architecture
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.embed_dim = self.model.config.hidden_size
            self.n_layers = self.model.config.num_hidden_layers
            print(f"  ViT-IN21k model: embed_dim={self.embed_dim}, n_layers={self.n_layers}")
            
        else:
            # Fallback: try generic loading
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.embed_dim = self.model.config.hidden_size
            self.n_layers = self.model.config.num_hidden_layers
            print(f"  Generic model: embed_dim={self.embed_dim}, n_layers={self.n_layers}")
        
        print(f"  Loaded successfully: embed_dim={self.embed_dim}, n_layers={self.n_layers}")
    
    def _pool_hidden_state(self, hidden_state):
        # hidden_state shape: [B, seq_len, D]
        # For DINOv3 with register tokens, CLS is still at index 0
        # For DeiT with distillation token, CLS is at index 0, distill at index 1
        if self.pooling == "cls":
            return hidden_state[:, 0]  # CLS token at index 0
        else:  # mean pooling
            return hidden_state.mean(dim=1)
    
    def forward(self, x, return_layers=False):
        # x is already preprocessed tensor [B, 3, H, W]
        
        # Run model with hidden states output
        outputs = self.model(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of [B, seq_len, D]
        
        # Debug print for first forward pass
        if not hasattr(self, '_debug_printed'):
            print(f"  Forward pass: {len(hidden_states)} hidden states, shape={hidden_states[0].shape}")
            self._debug_printed = True
        
        if return_layers:
            # Return pooled features from each layer
            # hidden_states[0] is the embedding layer output (before transformer blocks)
            # hidden_states[1:] are the transformer block outputs
            # We return n_layers features (skipping embedding layer at index 0)
            feats = []
            for hs in hidden_states[1:]:  # Skip initial embedding layer
                feats.append(self._pool_hidden_state(hs))
            return feats
        else:
            # Return final layer pooled features
            return self._pool_hidden_state(hidden_states[-1])

# Dual encoder for DINO+MAE ablation (keeping for compatibility)
class DualHFViTEncoder(nn.Module):
    def __init__(self, dino_name, mae_name, img_size=224):
        super().__init__()
        self.dino = HFViTEncoder(dino_name, img_size)
        self.mae = HFViTEncoder(mae_name, img_size)
        
        self.n_dino_layers = self.dino.n_layers
        self.n_mae_layers = self.mae.n_layers
        self.total_layers = self.n_dino_layers + self.n_mae_layers
        
        self.dino_indices = list(range(self.n_dino_layers))
        self.mae_indices = list(range(self.n_dino_layers, self.total_layers))
        
        self.dino_dim = self.dino.embed_dim
        self.mae_dim = self.mae.embed_dim
    
    def forward(self, x, return_layers=False):
        dino_feats = self.dino(x, return_layers=return_layers)
        mae_feats = self.mae(x, return_layers=return_layers)
        
        if return_layers:
            return dino_feats + mae_feats
        else:
            return torch.cat([dino_feats, mae_feats], dim=-1)
    
    def get_layer_info(self, idx):
        if idx < self.n_dino_layers:
            return 'dino', idx, self.dino_dim
        else:
            return 'mae', idx - self.n_dino_layers, self.mae_dim


class HFDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, split, img_size=224):
        if dataset_name == "saberzl/SID_Set":
            self.ds = load_from_disk(str(LOCAL_DS_DIR_SIDA))
            self.ds = self.ds[split]
        else:
            self.ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
        f = self.ds.features

        if "image" in f: self.img_key = "image"
        elif "img" in f: self.img_key = "img"
        else: raise KeyError(f"No image column found. Available: {list(f.keys())}")

        if "label" in f: self.label_key = "label"
        elif "fine_label" in f: self.label_key = "fine_label"
        else: raise KeyError(f"No label column found. Available: {list(f.keys())}")

        self.tf = v2.Compose([
            v2.Resize(img_size),
            v2.CenterCrop(img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _normalize_label(self, y):
        if isinstance(y, bool):
            return int(y)
        if isinstance(y, str):
            if y.lower() == "true": return 1
            if y.lower() == "false": return 0
        return int(y)

    def __getitem__(self, i):
        x = self.ds[i]
        y = self._normalize_label(x[self.label_key])
        return self.tf(x[self.img_key].convert("RGB")), y

    def __len__(self):
        return len(self.ds)


# def compute_isotropy(X, eps=1e-9):
#     Xc = X - X.mean(0, keepdim=True)
#     eigs = torch.linalg.eigvalsh((Xc.t() @ Xc) / Xc.shape[0]).real.clamp(min=0.0)
#     return (eigs.mean() / (eigs.std(unbiased=False) + eps)).item()

def compute_isotropy(X, eps=1e-6):
    Xc = X - X.mean(0, keepdim=True)
    n_samples, n_features = Xc.shape
    
    # For very high dimensional features, use SVD instead of eigendecomposition
    # SVD is more numerically stable
    if n_features > 512:
        try:
            # Use SVD: singular values squared / n_samples = eigenvalues of covariance
            U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
            eigs = (S ** 2) / n_samples
            eigs = eigs.clamp(min=eps)
            return (eigs.mean() / (eigs.std(unbiased=False) + eps)).item()
        except:
            print("  Warning: SVD failed, returning default isotropy=1.0")
            return 1.0
    else:
        cov = (Xc.t() @ Xc) / n_samples
        cov = cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
        try:
            eigs = torch.linalg.eigvalsh(cov).real.clamp(min=0.0)
            return (eigs.mean() / (eigs.std(unbiased=False) + eps)).item()
        except:
            print("  Warning: eigvalsh failed, returning default isotropy=1.0")
            return 1.0

def closed_form_ridge(X, Y, reg=1e-3):
    Xc, Yc = X - X.mean(0, keepdim=True), Y - Y.mean(0, keepdim=True)
    W = torch.linalg.solve(Xc.t() @ Xc + reg * torch.eye(X.shape[1], device=X.device), Xc.t() @ Yc)
    b = (Y.mean(0, keepdim=True) - X.mean(0, keepdim=True) @ W).squeeze(0)
    return W, b

def collect_calibration_embeddings(net, dataset, n_cal, batch_size, device="cuda"):
    net.eval()
    idx = random.sample(range(len(dataset)), n_cal) if n_cal < len(dataset) else list(range(len(dataset)))
    loader = DataLoader(Subset(dataset, idx), batch_size, shuffle=False, num_workers=4)
    embeddings, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            f = [t.cpu() for t in net(x.to(device), return_layers=True)]
            if not embeddings: embeddings = [[] for _ in f]
            for i, t in enumerate(f): embeddings[i].append(t)
            labels.append(y)
            if sum(len(b) for b in labels) >= n_cal: break
    return [torch.cat(e)[:n_cal] for e in embeddings], torch.cat(labels)[:n_cal].long()

def loes_select_layers(embeddings, labels, K, reg=1e-3, alpha=1.0, gamma=0.5, eta=0.1):
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
            B_orth = torch.linalg.solve(XS_c.t() @ XS_c + 1e-6 * torch.eye(XS_c.shape[1], device=X.device), XS_c.t() @ Xc)
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
                a, b, c_pt = cents[idx[:,0]], cents[idx[:,1]], cents[idx[:,2]]
                ab, ac = a-b, a-c_pt
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

def compute_all_layer_scores(embeddings, labels, reg=1e-3, alpha=1.0):
    Y = F.one_hot(labels, int(labels.max())+1).float()
    
    layer_scores = []
    for i, X in enumerate(embeddings):
        W, b = closed_form_ridge(X, Y, reg)
        loss = ((X @ W + b - Y)**2).mean().item()
        iso = compute_isotropy(X)
        score = loss + alpha*(1-iso)
        
        layer_scores.append({
            'layer_idx': i,
            'loes_score': score,
            'classification_loss': loss,
            'isotropy': iso,
            'embed_dim': X.shape[1]
        })
    
    return layer_scores


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
        a, b, c = centroids[idx[0]], centroids[idx[1]], centroids[idx[2]]
        ab, ac = a - b, a - c
        area = 0.5 * torch.sqrt((ab.pow(2).sum() * ac.pow(2).sum() - (ab * ac).sum().pow(2)).clamp(min=1e-6))
        cov = torch.cov(feats.T) + 1e-4 * torch.eye(feats.shape[1], device=feats.device)
        try: iso_loss = torch.linalg.eigvalsh(cov).real.clamp(min=1e-6).var()
        except: return torch.tensor(0.0, device=feats.device)
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
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            f = net(x, return_layers=True)
            sel = [f[i] for i in topk]
            if adapters: sel = [adapters[i](sf) for i, sf in enumerate(sel)]
            if aggregator: emb = aggregator(sel)
            elif fusion == "concat": emb = torch.cat(sel, dim=-1)
            elif fusion == "mean": emb = torch.stack(sel, dim=0).mean(0)
            elif fusion == "sum": emb = torch.stack(sel, dim=0).sum(0)
            corr += (probe(emb).argmax(1)==y).sum().item()
            tot += y.shape[0]
    return corr / tot if tot > 0 else 0.0


def save_layer_scores_csv(layer_scores, model_tag, dataset_name, output_path):
    df = pd.DataFrame(layer_scores)
    df.insert(0, 'model', model_tag)
    df.insert(1, 'dataset', dataset_name)
    df.to_csv(output_path, index=False)
    print(f"  Saved layer scores to {output_path}")


def plot_layer_quality_curve(layer_scores, model_tag, dataset_name, output_path):
    plt.figure(figsize=(10, 6))
    
    layer_indices = [s['layer_idx'] for s in layer_scores]
    loes_scores = [s['loes_score'] for s in layer_scores]
    
    plt.plot(layer_indices, loes_scores, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('LOES Score (lower = better)', fontsize=12)
    plt.title(f'{model_tag} - Layer Quality on {dataset_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved graph to {output_path}")


def run_phase1_scoring(cfg, device):
    output_dir = Path(cfg.multi_model_analysis.output_dir)
    scores_dir = output_dir / cfg.multi_model_analysis.per_layer_scores_dir
    graphs_dir = output_dir / cfg.multi_model_analysis.per_layer_graphs_dir
    
    dataset_name = cfg.dataset.name
    dataset_safe = dataset_name.split('/')[-1]
    train_ds = HFDataset(dataset_name, cfg.dataset.train_split, cfg.model.img_size)
    
    (scores_dir / dataset_safe).mkdir(parents=True, exist_ok=True)
    (graphs_dir / dataset_safe).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"PHASE 1: SCORING ALL LAYERS - Dataset: {dataset_safe}")
    print("="*80)
    
    all_model_results = {}
    
    for model_cfg in cfg.multi_model_analysis.models:
        model_name = model_cfg['name']
        model_tag = model_cfg['tag']
        
        print(f"\nAnalyzing {model_tag} ({model_name})")
        
        # Load HuggingFace model
        net = HFViTEncoder(model_name, cfg.model.img_size).to(device)
        for p in net.parameters(): 
            p.requires_grad = False
        
        # Collect embeddings
        
        n_cal = int(len(train_ds) * cfg.calibration.n_cal_pct)
        embeddings, labels = collect_calibration_embeddings(
            net, train_ds, n_cal, cfg.calibration.cal_bs, device
        )
        
        # Compute scores for all layers
        layer_scores = compute_all_layer_scores(embeddings, labels)
        
        # Save CSV
        csv_path = scores_dir / dataset_safe / f"{model_tag}_layer_scores.csv"
        save_layer_scores_csv(layer_scores, model_tag, dataset_safe, csv_path)
        
        # Generate graph
        graph_path = graphs_dir / dataset_safe / f"{model_tag}_quality_curve.png"
        plot_layer_quality_curve(layer_scores, model_tag, dataset_safe, graph_path)
        
        # Get top-3 layers
        sorted_scores = sorted(layer_scores, key=lambda x: x['loes_score'])
        top_3_indices = [s['layer_idx'] for s in sorted_scores[:3]]
        avg_loes = sum(s['loes_score'] for s in sorted_scores[:3]) / 3
        
        all_model_results[model_tag] = {
            'top_3': top_3_indices,
            'avg_loes': avg_loes,
            'embed_dim': net.embed_dim
        }
        
        print(f"  Top-3 layers: {top_3_indices}")
        print(f"  Avg LOES score: {avg_loes:.4f}")
        
        # Cleanup
        del net, embeddings
        torch.cuda.empty_cache()
    
    # Print model rankings
    print("\n" + "="*80)
    print("MODEL RANKINGS (by avg top-3 LOES score)")
    print("="*80)
    
    rankings = sorted(all_model_results.items(), key=lambda x: x[1]['avg_loes'])
    ranking_data = []
    
    for rank, (model_tag, results) in enumerate(rankings, 1):
        print(f"{rank}. {model_tag:15s} - Score: {results['avg_loes']:.4f} - Layers: {results['top_3']}")
        ranking_data.append({
            'rank': rank,
            'model': model_tag,
            'dataset': dataset_safe,
            'top_3_layers': str(results['top_3']),
            'avg_loes_score': results['avg_loes']
        })
    
    # Save rankings CSV
    rankings_csv = output_dir / cfg.multi_model_analysis.model_rankings_csv
    df_rankings = pd.DataFrame(ranking_data)
    if rankings_csv.exists():
        df_rankings.to_csv(rankings_csv, mode='a', header=False, index=False)
    else:
        df_rankings.to_csv(rankings_csv, index=False)
    
    print(f"\nSaved rankings to {rankings_csv}")
    
    return all_model_results

def run_phase2_training(cfg, device, model_results):
    output_dir = Path(cfg.multi_model_analysis.output_dir)
    accuracies_csv = output_dir / cfg.multi_model_analysis.model_accuracies_csv
    
    dataset_name = cfg.dataset.name
    dataset_safe = dataset_name.split('/')[-1]
    train_ds = HFDataset(dataset_name, cfg.dataset.train_split, cfg.model.img_size)
    train_loader = DataLoader(train_ds, cfg.training.bs, shuffle=True, drop_last=True, num_workers=4)
    
    val_split = cfg.dataset.get("val_split")
    test_split = cfg.dataset.get("test_split")
    
    val_loader = None
    if val_split:
        val_ds = HFDataset(dataset_name, val_split, cfg.model.img_size)
        val_loader = DataLoader(val_ds, cfg.training.test_bs, num_workers=4)
        
    test_loader = None
    if test_split:
        test_ds = HFDataset(dataset_name, test_split, cfg.model.img_size)
        test_loader = DataLoader(test_ds, cfg.training.test_bs, num_workers=4)
    
    eval_loader = val_loader if val_loader else test_loader
    
    print("\n" + "="*80)
    print(f"PHASE 2: TRAINING WITH GREEDY LOES TOP-3 - Dataset: {dataset_safe}")
    print("="*80)
    
    results_data = []
    greedy_loes_scores = {}
    
    for model_cfg in cfg.multi_model_analysis.models:
        model_name = model_cfg['name']
        model_tag = model_cfg['tag']
        
        print(f"\nTraining {model_tag}")
        
        # Load HuggingFace model
        net = HFViTEncoder(model_name, cfg.model.img_size).to(device)
        for p in net.parameters(): 
            p.requires_grad = False
        
        # USE GREEDY LOES SELECTION
        print(f"  Running greedy LOES selection...")
        n_cal = int(len(train_ds) * cfg.calibration.n_cal_pct)
        embeddings, labels = collect_calibration_embeddings(
            net, train_ds, n_cal, cfg.calibration.cal_bs, device
        )
        
        # Use original greedy algorithm
        topk = loes_select_layers(embeddings, labels, K=3)
        
        # Compute average LOES score for these 3 layers
        Y = F.one_hot(labels, int(labels.max())+1).float()
        selected_scores = []
        for idx in topk:
            X = embeddings[idx]
            W, b = closed_form_ridge(X, Y, reg=1e-3)
            loss = ((X @ W + b - Y)**2).mean().item()
            iso = compute_isotropy(X)
            score = loss + 1.0*(1-iso)
            selected_scores.append(score)
        
        avg_greedy_loes = sum(selected_scores) / len(selected_scores)
        greedy_loes_scores[model_tag] = avg_greedy_loes
        
        print(f"  Selected layers (greedy): {topk}")
        print(f"  Avg LOES score: {avg_greedy_loes:.4f}")
        
        # Cleanup embeddings
        del embeddings, labels
        torch.cuda.empty_cache()
        
        # WandB init
        run_name = f"{model_tag}_top3greedy_{dataset_safe}"
        wandb.init(
            project=cfg.wandb.project, 
            name=run_name,
            config={
                'model': model_tag,
                'dataset': dataset_safe,
                'selected_layers': topk,
                'avg_greedy_loes_score': avg_greedy_loes,
                'phase': 'greedy_loes_top3',
                'selection_method': 'greedy_complementary'
            },
            reinit=True
        )
        
        # Create adapters and probe
        proj_dim = cfg.model.proj_dim
        adapter_list = [
            nn.Sequential(
                nn.LayerNorm(net.embed_dim), 
                nn.Linear(net.embed_dim, proj_dim), 
                nn.GELU()
            ) for _ in topk
        ]
        adapters = nn.ModuleList(adapter_list).to(device)
        
        fused_dim = proj_dim * len(topk)
        num_classes = len(set(i[train_ds.label_key] for i in train_ds.ds))
        probe = nn.Sequential(
            nn.LayerNorm(fused_dim), 
            nn.Dropout(0.2), 
            nn.Linear(fused_dim, num_classes)
        ).to(device)
        
        # Optimizer
        params = [
            {"params": probe.parameters(), "lr": cfg.optim.lr_probe},
            {"params": adapters.parameters(), "lr": cfg.optim.lr_probe}
        ]
        opt = torch.optim.AdamW(params, weight_decay=1e-4)
        sched = CosineAnnealingLR(opt, len(train_loader)*cfg.training.epochs, eta_min=1e-6)
        
        # Training loop
        best_acc = 0.0
        for ep in range(cfg.training.epochs):
            net.eval()
            probe.train()
            
            for x, y in tqdm.tqdm(train_loader, desc=f"Ep {ep}"):
                x, y = x.to(device), y.to(device)
                feats = net(x, return_layers=True)
                sel_feats = [feats[i] for i in topk]
                sel_feats = [adapters[i](f) for i, f in enumerate(sel_feats)]
                emb = torch.cat(sel_feats, dim=-1)
                
                loss = F.cross_entropy(probe(emb), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                
                wandb.log({"train_loss": loss.item()})
            
            # Evaluate
            acc = run_eval(eval_loader, net, probe, adapters, None, topk, "concat", device)
            if acc > best_acc:
                best_acc = acc
            
            wandb.log({"val_acc": acc, "epoch": ep})
            print(f"  Ep {ep} Acc: {acc:.4f}")
        
        # Final test
        final_test_acc = best_acc
        if val_loader and test_loader:
            final_test_acc = run_eval(test_loader, net, probe, adapters, None, topk, "concat", device)
            wandb.log({"final_test_acc": final_test_acc})
        
        wandb.finish()
        
        # Save results
        results_data.append({
            'model': model_tag,
            'dataset': dataset_safe,
            'phase': 'greedy_loes_top3',
            'selected_layers': str(topk),
            'avg_greedy_loes_score': avg_greedy_loes,
            'val_acc': best_acc,
            'test_acc': final_test_acc
        })
        
        print(f"  Best Acc: {best_acc:.4f}")
        
        # Cleanup
        del net, probe, adapters
        torch.cuda.empty_cache()
    
    # Print model rankings
    print("\n" + "="*80)
    print("MODEL RANKINGS (by greedy LOES top-3 avg score)")
    print("="*80)
    
    rankings = sorted(greedy_loes_scores.items(), key=lambda x: x[1])
    for rank, (model_tag, score) in enumerate(rankings, 1):
        result = next(r for r in results_data if r['model'] == model_tag)
        print(f"{rank}. {model_tag:15s} - LOES: {score:.4f} - Acc: {result['val_acc']:.2f}% - Layers: {result['selected_layers']}")
    
    # Save accuracies
    df_results = pd.DataFrame(results_data)
    if accuracies_csv.exists():
        df_results.to_csv(accuracies_csv, mode='a', header=False, index=False)
    else:
        df_results.to_csv(accuracies_csv, index=False)
    
    print(f"\nSaved accuracies to {accuracies_csv}")


def run_phase3_lastlayer(cfg, device):
    output_dir = Path(cfg.multi_model_analysis.output_dir)
    baseline_csv = output_dir / cfg.multi_model_analysis.last_layer_baseline_csv
    
    dataset_name = cfg.dataset.name
    dataset_safe = dataset_name.split('/')[-1]
    train_ds = HFDataset(dataset_name, cfg.dataset.train_split, cfg.model.img_size)
    train_loader = DataLoader(train_ds, cfg.training.bs, shuffle=True, drop_last=True, num_workers=4)
    
    val_split = cfg.dataset.get("val_split")
    test_split = cfg.dataset.get("test_split")
    
    val_loader = None
    if val_split:
        val_ds = HFDataset(dataset_name, val_split, cfg.model.img_size)
        val_loader = DataLoader(val_ds, cfg.training.test_bs, num_workers=4)
        
    test_loader = None
    if test_split:
        test_ds = HFDataset(dataset_name, test_split, cfg.model.img_size)
        test_loader = DataLoader(test_ds, cfg.training.test_bs, num_workers=4)
    
    eval_loader = val_loader if val_loader else test_loader
    
    print("\n" + "="*80)
    print(f"PHASE 3: LAST LAYER BASELINE - Dataset: {dataset_safe}")
    print("="*80)
    
    baseline_data = []
    
    for model_cfg in cfg.multi_model_analysis.models:
        model_name = model_cfg['name']
        model_tag = model_cfg['tag']
        
        print(f"\nTraining {model_tag} (last layer only)")
        
        # Load HuggingFace model
        net = HFViTEncoder(model_name, cfg.model.img_size).to(device)
        for p in net.parameters(): 
            p.requires_grad = False
        
        # Get last layer index
        total_layers = net.n_layers
        topk = [total_layers - 1]
        
        print(f"  Using layer: {topk[0]}")
        
        # WandB init
        run_name = f"{model_tag}_lastlayer_{dataset_safe}"
        wandb.init(
            project=cfg.wandb.project, 
            name=run_name,
            config={
                'model': model_tag,
                'dataset': dataset_safe,
                'layer': topk[0],
                'phase': 'last_layer_baseline'
            },
            reinit=True
        )
        
        # Create adapter and probe
        proj_dim = cfg.model.proj_dim
        adapters = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(net.embed_dim), 
                nn.Linear(net.embed_dim, proj_dim), 
                nn.GELU()
            )
        ]).to(device)
        
        num_classes = len(set(i[train_ds.label_key] for i in train_ds.ds))
        probe = nn.Sequential(
            nn.LayerNorm(proj_dim), 
            nn.Dropout(0.2), 
            nn.Linear(proj_dim, num_classes)
        ).to(device)
        
        # Optimizer
        params = [
            {"params": probe.parameters(), "lr": cfg.optim.lr_probe},
            {"params": adapters.parameters(), "lr": cfg.optim.lr_probe}
        ]
        opt = torch.optim.AdamW(params, weight_decay=1e-4)
        sched = CosineAnnealingLR(opt, len(train_loader)*cfg.training.epochs, eta_min=1e-6)
        
        # Training loop
        best_acc = 0.0
        for ep in range(cfg.training.epochs):
            net.eval()
            probe.train()
            
            for x, y in tqdm.tqdm(train_loader, desc=f"Ep {ep}"):
                x, y = x.to(device), y.to(device)
                feats = net(x, return_layers=True)
                sel_feats = [feats[topk[0]]]
                sel_feats = [adapters[0](sel_feats[0])]
                emb = sel_feats[0]
                
                loss = F.cross_entropy(probe(emb), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                
                wandb.log({"train_loss": loss.item()})
            
            # Evaluate
            acc = run_eval(eval_loader, net, probe, adapters, None, topk, "concat", device)
            if acc > best_acc:
                best_acc = acc
            
            wandb.log({"val_acc": acc, "epoch": ep})
            print(f"  Ep {ep} Acc: {acc:.4f}")
        
        # Final test
        final_test_acc = best_acc
        if val_loader and test_loader:
            final_test_acc = run_eval(test_loader, net, probe, adapters, None, topk, "concat", device)
            wandb.log({"final_test_acc": final_test_acc})
        
        wandb.finish()
        
        # Save results
        baseline_data.append({
            'model': model_tag,
            'dataset': dataset_safe,
            'layer_idx': topk[0],
            'val_acc': best_acc,
            'test_acc': final_test_acc
        })
        
        print(f"  Best Acc: {best_acc:.4f}")
        
        # Cleanup
        del net, probe, adapters
        torch.cuda.empty_cache()
    
    # Save baseline results
    df_baseline = pd.DataFrame(baseline_data)
    if baseline_csv.exists():
        df_baseline.to_csv(baseline_csv, mode='a', header=False, index=False)
    else:
        df_baseline.to_csv(baseline_csv, index=False)
    
    print(f"\nSaved baseline to {baseline_csv}")


def run_multi_model_analysis(cfg, device):
    print("\n" + "="*80)
    print("STARTING MULTI-MODEL ANALYSIS")
    print("="*80)
    print(f"Dataset: {cfg.dataset.name}")
    print(f"Models: {len(cfg.multi_model_analysis.models)}")
    print(f"Output: {cfg.multi_model_analysis.output_dir}")
    print("="*80)
    
    # Phase 1: Score all layers
    model_results = run_phase1_scoring(cfg, device)
    
    # Phase 2: Train top-3 layers
    run_phase2_training(cfg, device, model_results)
    
    # Phase 3: Last layer baseline
    run_phase3_lastlayer(cfg, device)
    
    print("\n" + "="*80)
    print("MULTI-MODEL ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Results saved to: {cfg.multi_model_analysis.output_dir}")


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = cfg.device
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    
    if cfg.multi_model_analysis.enabled:
        print("\n" + "="*80)
        print(" MULTI-MODEL ANALYSIS")
        print("="*80)
        run_multi_model_analysis(cfg, device)
        return

    # Single model training path (for dual model ablation)
    if cfg.model.use_dual:
        net = DualHFViTEncoder(
            cfg.model.dino_name, 
            cfg.model.mae_name, 
            cfg.model.img_size
        ).to(device)
        model_tag = "DINO+MAE"
    else:
        net = HFViTEncoder(cfg.model.name, cfg.model.img_size).to(device)
        model_tag = cfg.model.name

    if not cfg.optim.finetune:
        for p in net.parameters(): p.requires_grad = False

    ablation_tag = ""
    if cfg.ablation.no_adapters: ablation_tag += "_NoAdapt"
    if not cfg.ablation.use_geo_loss: ablation_tag += "_NoGeo"
    if cfg.ablation.fusion != "concat": ablation_tag += f"_{cfg.ablation.fusion}"
    run_name = f"{model_tag}_{cfg.selection.mode}_k{cfg.topk}{ablation_tag}"
    wandb.init(project=cfg.wandb.project, name=run_name, config=dict(cfg))

    train_ds = HFDataset(cfg.dataset.name, cfg.dataset.train_split, cfg.model.img_size)
    train_loader = DataLoader(train_ds, cfg.training.bs, shuffle=True, drop_last=True, num_workers=4)

    val_split = cfg.dataset.get("val_split")
    test_split = cfg.dataset.get("test_split")

    val_loader = None
    if val_split:
        val_ds = HFDataset(cfg.dataset.name, val_split, cfg.model.img_size)
        val_loader = DataLoader(val_ds, cfg.training.test_bs, num_workers=4)
        
    test_loader = None
    if test_split:
        test_ds = HFDataset(cfg.dataset.name, test_split, cfg.model.img_size)
        test_loader = DataLoader(test_ds, cfg.training.test_bs, num_workers=4)

    if not cfg.optim.finetune:
        for p in net.parameters(): p.requires_grad = False
        
        
    n_cal = int(len(train_ds) * cfg.calibration.n_cal_pct)

    embeddings, labels = collect_calibration_embeddings(net, train_ds, n_cal, cfg.calibration.cal_bs, device)
    total_layers = len(embeddings)
    
    if cfg.ablation.fusion == "mean": 
        topk = list(range(total_layers-3, total_layers))
    elif cfg.selection.mode == "loes": 
        topk = loes_select_layers(embeddings, labels, K=cfg.topk)
    elif cfg.selection.mode == "random": 
        topk = random.sample(range(total_layers), cfg.topk)
    elif cfg.selection.mode == "last":
        if cfg.model.use_dual:
            topk = [net.n_dino_layers - 1, net.total_layers - 1]
            print(f"Dual baseline: Using last layers from both models: {topk}")
        else:
            topk = [total_layers - 1]
    elif cfg.selection.mode in ["concat_all", "avg_all", "learnable_weight"]: 
        topk = list(range(total_layers))
    else: 
        raise ValueError(f"Unknown mode {cfg.selection.mode}")
    
    print(f"Selected Layers: {topk}")
    if cfg.model.use_dual:
        for idx in topk:
            model_name, local_idx, dim = net.get_layer_info(idx)
            print(f"  Global idx {idx} -> {model_name.upper()} layer {local_idx} (dim={dim})")
        
        dino_count = sum(1 for i in topk if i < net.n_dino_layers)
        mae_count = len(topk) - dino_count
        print(f"  Summary: {dino_count} DINO layers, {mae_count} MAE layers")
    proj_dim = cfg.model.proj_dim if (not cfg.ablation.no_adapters) else None

    if not cfg.ablation.no_adapters:
        adapter_list = []
        for layer_idx in topk:
            if cfg.model.use_dual:
                model_name, local_idx, embed_dim = net.get_layer_info(layer_idx)
            else:
                embed_dim = net.embed_dim
            
            adapter_list.append(
                nn.Sequential(
                    nn.LayerNorm(embed_dim), 
                    nn.Linear(embed_dim, proj_dim), 
                    nn.GELU()
                )
            )
        adapters = nn.ModuleList(adapter_list).to(device)
    else:
        adapters = None
        if cfg.model.use_dual:
            proj_dim = sum(net.get_layer_info(i)[2] for i in topk) if cfg.ablation.fusion == "concat" else net.dino_dim
        else:
            proj_dim = net.embed_dim
    
    aggregator = None
    if cfg.selection.mode == "learnable_weight":
        aggregator = LearnableWeighting(len(topk)).to(device)
        fused_dim = proj_dim if not cfg.ablation.no_adapters else proj_dim
    elif cfg.ablation.fusion == "concat":
        if cfg.ablation.no_adapters and cfg.model.use_dual:
            fused_dim = sum(net.get_layer_info(i)[2] for i in topk)
        else:
            fused_dim = proj_dim * len(topk)
    else:
        fused_dim = proj_dim if not cfg.ablation.no_adapters else (net.get_layer_info(topk[0])[2] if cfg.model.use_dual else net.embed_dim)

    num_classes = len(set(i[train_ds.label_key] for i in train_ds.ds))
    probe = nn.Sequential(nn.LayerNorm(fused_dim), nn.Dropout(0.2), nn.Linear(fused_dim, num_classes)).to(device)

    params = [{"params": probe.parameters(), "lr": cfg.optim.lr_probe}]
    if adapters: params.append({"params": adapters.parameters(), "lr": cfg.optim.lr_probe})
    if aggregator: params.append({"params": aggregator.parameters(), "lr": 1e-3})
    if cfg.optim.finetune: params.append({"params": net.parameters(), "lr": cfg.optim.lr_backbone})
    
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, len(train_loader)*cfg.training.epochs, eta_min=1e-6)
    geo_loss = GeometricLoss(weight=0.1 if cfg.ablation.use_geo_loss else 0.0)

    best_acc = 0.0
    ckpt_path = "best_model.pth"
    eval_loader = val_loader if val_loader else test_loader
    
    for ep in range(cfg.training.epochs):
        net.train() if cfg.optim.finetune else net.eval()
        probe.train()
        for x, y in tqdm.tqdm(train_loader, desc=f"Ep {ep}"):
            x, y = x.to(device), y.to(device)
            feats = net(x, return_layers=True)
            sel_feats = [feats[i] for i in topk]
            if adapters: sel_feats = [adapters[i](f) for i, f in enumerate(sel_feats)]
            
            if aggregator: emb = aggregator(sel_feats)
            elif cfg.ablation.fusion == "concat": emb = torch.cat(sel_feats, dim=-1)
            elif cfg.ablation.fusion == "mean": emb = torch.stack(sel_feats, dim=0).mean(0)
            elif cfg.ablation.fusion == "sum": emb = torch.stack(sel_feats, dim=0).sum(0)
            
            loss = F.cross_entropy(probe(emb), y) + geo_loss(emb, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            wandb.log({"train_loss": loss.item()})
        
        acc = run_eval(eval_loader, net, probe, adapters, aggregator, topk, cfg.ablation.fusion, device)
        if acc > best_acc:
            best_acc = acc
            state = {'probe': probe.state_dict()}
            if adapters: state['adapters'] = adapters.state_dict()
            if aggregator: state['aggregator'] = aggregator.state_dict()
            if cfg.optim.finetune: state['net'] = net.state_dict()
            torch.save(state, ckpt_path)
            
        wandb.log({"val_acc": acc})
        print(f"Ep {ep} Acc: {acc:.4f}")

    final_test_acc = best_acc
    if val_loader and test_loader:
        print("Loading best model for test evaluation...")
        ckpt = torch.load(ckpt_path)
        probe.load_state_dict(ckpt['probe'])
        if adapters: adapters.load_state_dict(ckpt['adapters'])
        if aggregator: aggregator.load_state_dict(ckpt['aggregator'])
        if cfg.optim.finetune: net.load_state_dict(ckpt['net'])
        
        final_test_acc = run_eval(test_loader, net, probe, adapters, aggregator, topk, cfg.ablation.fusion, device)
        print(f"Final Test Acc: {final_test_acc:.4f}")
        wandb.log({"final_test_acc": final_test_acc})

    if cfg.model.use_dual:
        dino_layers = [i for i in topk if i < net.n_dino_layers]
        mae_layers = [i - net.n_dino_layers for i in topk if i >= net.n_dino_layers]
        model_info = f"DINO+MAE"
        layer_breakdown = f"D{len(dino_layers)}_M{len(mae_layers)}"
    else:
        model_info = cfg.model.name
        layer_breakdown = "single"

    row = {
        "run": run_name, 
        "model": model_info,
        "mode": cfg.selection.mode, 
        "k": cfg.topk,
        "layer_breakdown": layer_breakdown,
        "selected_layers": str(topk),
        "fusion": cfg.ablation.fusion, 
        "geo_loss": cfg.ablation.use_geo_loss,
        "adapters": not cfg.ablation.no_adapters, 
        "val_acc": best_acc, 
        "test_acc": final_test_acc,
        "dataset": cfg.dataset.name
    }
    with open(cfg.logging.results_csv, "a") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if f.tell()==0: w.writeheader()
        w.writerow(row)
    
    if os.path.exists(ckpt_path): os.remove(ckpt_path)


if __name__ == "__main__": 
    main()