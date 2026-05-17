import random
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import wandb
import hydra
import tqdm
import numpy as np
from omegaconf import DictConfig
from datasets import load_dataset, Audio
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# ==========================================
# 1. MODEL WRAPPER (Updated for Wav2Vec2)
# ==========================================
class Wav2Vec2Encoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base-960h", num_labels=10):
        super().__init__()
        # Load HF Model with hidden states enabled
        self.backbone = AutoModelForAudioClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            output_hidden_states=True,
            ignore_mismatched_sizes=True
        )
        self.config = self.backbone.config
        self.embed_dim = self.config.hidden_size

    def forward(self, x, return_layers=False):
        # x shape: [Batch, SequenceLength]
        outputs = self.backbone(x)
        
        # If we just want the final prediction (standard forward pass)
        if not return_layers:
            return outputs.logits

        # If we want intermediate layers for LOES
        # Wav2Vec2 hidden_states is a tuple of (Batch, Time, Dim) tensors
        # We must MEAN POOL over the Time dimension to get (Batch, Dim) vectors for LOES
        hidden_states = outputs.hidden_states 
        
        pooled_feats = []
        for state in hidden_states:
            # state: [B, T, D] -> mean(dim=1) -> [B, D]
            pooled_feats.append(state.mean(dim=1))
            
        return pooled_feats

# ==========================================
# 2. DATASET WRAPPER (Updated for Audio)
# ==========================================
import torch
import numpy as np
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor

class AudioHFDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, split, model_name, max_seconds=3.0):
        # 1. Handle dataset name aliasing
        if "gtzan" in dataset_name:
            dataset_name = "marsyas/gtzan"
        elif "english_accents" in dataset_name and "/" not in dataset_name:
            dataset_name = "alexnasa/english_accents"
            
        print(f"Loading dataset: {dataset_name} | Requested split: {split}")
        
        # 2. Load the dataset (GTZAN requires specific handling)
        if dataset_name == "marsyas/gtzan":
            # GTZAN only has a 'train' split generally. We must manually split it.
            full_ds = load_dataset(dataset_name, "all", split="train")
            
            # Create a 90/10 split deterministically (Seed 42)
            # The 'file' column is a good hashable key for splitting if needed, 
            # but HF's train_test_split is easier.
            splits = full_ds.train_test_split(test_size=0.1, seed=42)
            
            if split == "train":
                self.ds = splits["train"]
            elif split == "test" or split == "validation":
                self.ds = splits["test"]
            else:
                raise ValueError(f"GTZAN only supports 'train' or 'test' (derived via split). Got {split}")
                
            self.label_key = "genre" # GTZAN uses 'genre'
            
        else:
            # Standard Loading for other datasets
            self.ds = load_dataset(dataset_name, split=split)
            self.label_key = "region" if "english_accents" in dataset_name else "label"
            
            # Encode labels if they are strings
            if isinstance(self.ds.features[self.label_key],  type(None)) or hasattr(self.ds.features[self.label_key], 'names'):
                 pass # Already ClassLabel or Int
            else:
                 self.ds = self.ds.class_encode_column(self.label_key)

        # 3. Audio Setup
        self.audio_key = "audio"
        self.sampling_rate = 16000
        self.ds = self.ds.cast_column(self.audio_key, Audio(sampling_rate=self.sampling_rate))
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.max_length = int(self.sampling_rate * max_seconds)

    def __getitem__(self, i):
        item = self.ds[i]
        audio_array = item[self.audio_key]["array"]
        
        # Safety for empty audio
        if len(audio_array) == 0:
            audio_array = np.random.uniform(-0.01, 0.01, 16000)
            
        # Processor handles padding/truncation
        inputs = self.processor(
            audio_array, 
            sampling_rate=self.sampling_rate, 
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Handle Label (GTZAN labels might need int casting)
        label = item[self.label_key]
        
        return inputs.input_values.squeeze(0), label

    def __len__(self):
        return len(self.ds)
# ==========================================
# 3. LOES ALGORITHMS (Unchanged Logic)
# ==========================================
def compute_isotropy(X, eps=1e-9):
    Xc = X - X.mean(0, keepdim=True)
    eigs = torch.linalg.eigvalsh((Xc.t() @ Xc) / Xc.shape[0]).real.clamp(min=0.0)
    return (eigs.mean() / (eigs.std(unbiased=False) + eps)).item()

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
            x = x.to(device)
            # Returns list of [B, D] tensors
            f = [t.cpu() for t in net(x, return_layers=True)]
            if not embeddings: embeddings = [[] for _ in f]
            for i, t in enumerate(f): embeddings[i].append(t)
            labels.append(y)
            if sum(len(b) for b in labels) >= n_cal: break
            
    return [torch.cat(e)[:n_cal].to(device) for e in embeddings], torch.cat(labels)[:n_cal].long().to(device)

def loes_select_layers(embeddings, labels, K, reg=1e-3, alpha=1.0, gamma=0.5, eta=0.1):
    # Ensure inputs are on the same device
    device = embeddings[0].device
    Y = F.one_hot(labels, int(labels.max())+1).float().to(device)
    
    best = (float("inf"), -1, None)
    
    # 1. Select First Layer
    for i, X in enumerate(embeddings):
        W, b = closed_form_ridge(X, Y, reg)
        loss = ((X @ W + b - Y)**2).mean().item()
        iso = compute_isotropy(X)
        score = loss + alpha*(1-iso)
        if score < best[0]: best = (score, i, (W, b))
    
    selected = [best[1]]
    X_S = embeddings[best[1]].clone()
    y_hat = embeddings[best[1]] @ best[2][0] + best[2][1]
    residual = Y - y_hat
    
    # 2. Select Subsequent Layers
    while len(selected) < K:
        best = (float("inf"), None, None)
        for i, X in enumerate(embeddings):
            if i in selected: continue
            
            # Orthogonalize Candidate X against Selected X_S
            Xc, XS_c = X - X.mean(0, keepdim=True), X_S - X_S.mean(0, keepdim=True)
            
            # Solve XS_c * B = Xc
            B_orth = torch.linalg.solve(XS_c.t() @ XS_c + 1e-6 * torch.eye(XS_c.shape[1], device=device), XS_c.t() @ Xc)
            X_tilde = Xc - XS_c @ B_orth + X.mean(0, keepdim=True)
            
            W, b = closed_form_ridge(X_tilde, residual, reg)
            res_loss = ((X_tilde @ W + b - residual)**2).mean().item()
            iso = compute_isotropy(X)
            
            # Redundancy Term
            red = max([ (torch.norm(X.t()@embeddings[j])/(torch.norm(X)*torch.norm(embeddings[j]))).item() for j in selected])
            
            # Geometric Triangle Term
            classes = torch.unique(labels)
            cents = torch.stack([X_tilde[labels==c].mean(0) for c in classes]) if len(classes)>=3 else None
            tri = 0.0
            if cents is not None and cents.shape[0] >= 3:
                # Randomly sample triplets for speed
                idx_tri = torch.randint(0, len(cents), (min(200, len(cents)*3), 3), device=device)
                a, b_pt, c_pt = cents[idx_tri[:,0]], cents[idx_tri[:,1]], cents[idx_tri[:,2]]
                ab, ac = a-b_pt, a-c_pt
                tri = 0.5 * torch.sqrt((ab.pow(2).sum(1)*ac.pow(2).sum(1)-(ab*ac).sum(1).pow(2)).clamp(min=0)).mean().item()
            
            score = res_loss + alpha*(1-iso) + gamma*red - eta*tri
            if score < best[0]: best = (score, i, (W, b, X_tilde))
        
        if best[1] is None: break
        idx, (W, b, _) = best[1], best[2]
        
        # Update Residuals
        W_f, b_f = closed_form_ridge(embeddings[idx], residual + y_hat, reg)
        y_hat += embeddings[idx] @ W_f + b_f
        residual = Y - y_hat
        X_S = torch.cat([X_S, embeddings[idx]], dim=1)
        selected.append(idx)
        
    return selected

class GeometricLoss(nn.Module):
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
    def forward(self, feats, labels):
        if self.weight <= 0: return torch.tensor(0.0, device=feats.device)
        classes = torch.unique(labels)
        if len(classes) < 3: return torch.tensor(0.0, device=feats.device)
        
        centroids = []
        for c in classes:
            mask = labels == c
            if mask.sum() > 0:
                centroids.append(feats[mask].mean(0))
        centroids = torch.stack(centroids)
        
        if centroids.shape[0] < 3: return torch.tensor(0.0, device=feats.device)
        
        # Pick 3 random centroids
        idx = torch.randperm(len(centroids))[:3]
        a, b, c = centroids[idx[0]], centroids[idx[1]], centroids[idx[2]]
        ab, ac = a - b, a - c
        
        # Area of triangle formed by centroids
        area = 0.5 * torch.sqrt((ab.pow(2).sum() * ac.pow(2).sum() - (ab * ac).sum().pow(2)).clamp(min=1e-6))
        
        # Global Isotropy
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

# ==========================================
# 4. EVALUATION & MAIN LOOP
# ==========================================
def run_eval(loader, net, probe, adapters, aggregator, topk, fusion, device):
    net.eval(); probe.eval()
    corr, tot = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # Use return_layers=True to get pooled hidden states
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

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = cfg.device
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    
    ablation_tag = ""
    if cfg.ablation.no_adapters: ablation_tag += "_NoAdapt"
    if not cfg.ablation.use_geo_loss: ablation_tag += "_NoGeo"
    if cfg.ablation.fusion != "concat": ablation_tag += f"_{cfg.ablation.fusion}"
    
    run_name = f"AUDIO_{cfg.model.name}_{cfg.selection.mode}_k{cfg.topk}{ablation_tag}"
    # Sanitized name for wandb
    safe_name = run_name.replace("/", "_") 
    wandb.init(project=cfg.wandb.project, name=safe_name, config=dict(cfg))

    # Init Dataset (Audio Specific)
    # Note: max_seconds needs to be tuned for your dataset. 
    # Speech Commands ~ 1.0s, GTZAN ~ 30s (usually clipped to 3-5s for training)
    train_ds = AudioHFDataset(cfg.dataset.name, cfg.dataset.train_split, cfg.model.name, max_seconds=3.0)
    
    num_classes = len(set(train_ds.ds[train_ds.label_key]))
    
    train_loader = DataLoader(train_ds, cfg.training.bs, shuffle=True, drop_last=True, num_workers=4)
    
    val_loader = None
    if cfg.dataset.get("val_split"):
        val_ds = AudioHFDataset(cfg.dataset.name, cfg.dataset.val_split, cfg.model.name, max_seconds=3.0)
        val_loader = DataLoader(val_ds, cfg.training.test_bs, num_workers=4)
        
    test_loader = None
    if cfg.dataset.get("test_split"):
        test_ds = AudioHFDataset(cfg.dataset.name, cfg.dataset.test_split, cfg.model.name, max_seconds=3.0)
        test_loader = DataLoader(test_ds, cfg.training.test_bs, num_workers=4)

    # Init Model (Audio Specific)
    net = Wav2Vec2Encoder(cfg.model.name, num_labels=num_classes).to(device)
    
    if not cfg.optim.finetune:
        for p in net.parameters(): p.requires_grad = False

    print("Collecting Calibration Embeddings...")
    embeddings, labels = collect_calibration_embeddings(net, train_ds, cfg.calibration.n_cal, cfg.calibration.cal_bs, device)
    total_layers = len(embeddings)
    print(f"Total Layers found: {total_layers}")
    
    # Selection Logic
    if cfg.ablation.fusion == "mean": topk = list(range(total_layers-3, total_layers))
    elif cfg.selection.mode == "loes": topk = loes_select_layers(embeddings, labels, K=cfg.topk)
    elif cfg.selection.mode == "random": topk = random.sample(range(total_layers), cfg.topk)
    elif cfg.selection.mode == "last": topk = [total_layers-1]
    elif cfg.selection.mode == "learnable_weight": topk = list(range(total_layers))
    else: raise ValueError(f"Unknown mode {cfg.selection.mode}")
    
    print(f"Selected Layers: {topk}")
    
    D = net.embed_dim
    proj_dim = cfg.model.proj_dim if (not cfg.ablation.no_adapters) else D
    
    adapters = nn.ModuleList([nn.Sequential(nn.LayerNorm(D), nn.Linear(D, proj_dim), nn.GELU()) for _ in topk]).to(device) if not cfg.ablation.no_adapters else None
    
    aggregator = None
    if cfg.selection.mode == "learnable_weight":
        aggregator = LearnableWeighting(len(topk)).to(device)
        fused_dim = proj_dim
    elif cfg.ablation.fusion == "concat": fused_dim = proj_dim * len(topk)
    else: fused_dim = proj_dim

    probe = nn.Sequential(nn.LayerNorm(fused_dim), nn.Dropout(0.2), nn.Linear(fused_dim, num_classes)).to(device)

    # Optimization Setup
    params = [{"params": probe.parameters(), "lr": cfg.optim.lr_probe}]
    if adapters: params.append({"params": adapters.parameters(), "lr": cfg.optim.lr_probe})
    if aggregator: params.append({"params": aggregator.parameters(), "lr": 1e-5})
    if cfg.optim.finetune: params.append({"params": net.parameters(), "lr": cfg.optim.lr_backbone})
    
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, len(train_loader)*cfg.training.epochs, eta_min=1e-6)
    geo_loss = GeometricLoss(weight=0.1 if cfg.ablation.use_geo_loss else 0.0)

    best_acc = 0.0
    ckpt_path = "best_audio_model.pth"
    eval_loader = val_loader if val_loader else test_loader
    
    # Training Loop
    for ep in range(cfg.training.epochs):
        net.train() if cfg.optim.finetune else net.eval()
        probe.train()
        
        for x, y in tqdm.tqdm(train_loader, desc=f"Ep {ep}"):
            x, y = x.to(device), y.to(device)
            
            # Forward Pass: Get layers
            feats = net(x, return_layers=True)
            
            sel_feats = [feats[i] for i in topk]
            if adapters: sel_feats = [adapters[i](f) for i, f in enumerate(sel_feats)]
            
            if aggregator: emb = aggregator(sel_feats)
            elif cfg.ablation.fusion == "concat": emb = torch.cat(sel_feats, dim=-1)
            elif cfg.ablation.fusion == "mean": emb = torch.stack(sel_feats, dim=0).mean(0)
            elif cfg.ablation.fusion == "sum": emb = torch.stack(sel_feats, dim=0).sum(0)
            
            loss = F.cross_entropy(probe(emb), y) + geo_loss(emb, y)
            
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
            if adapters: torch.nn.utils.clip_grad_norm_(adapters.parameters(), max_norm=1.0)
            opt.step(); sched.step()
            wandb.log({"train_loss": loss.item()})
        
        # Eval
        if eval_loader:
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

    # Final Test
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

    # CSV Logging
    row = {
        "run": run_name, "mode": cfg.selection.mode, "k": cfg.topk,
        "fusion": cfg.ablation.fusion, "geo_loss": cfg.ablation.use_geo_loss,
        "adapters": not cfg.ablation.no_adapters, "val_acc": best_acc, "test_acc": final_test_acc , 'dataset': cfg.dataset.name
    }
    
    # Ensure csv directory exists
    csv_path = cfg.logging.results_csv
    
    # Only create a directory if the path actually HAS a directory component
    if os.path.dirname(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(cfg.logging.results_csv, "a") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if f.tell()==0: w.writeheader()
        w.writerow(row)
    
    if os.path.exists(ckpt_path): os.remove(ckpt_path)

if __name__ == "__main__": 
    main()
