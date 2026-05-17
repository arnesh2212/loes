import random
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["WANDB_MODE"] = "offline"

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


import os
from datasets import load_dataset, load_from_disk

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
            if adapters: sel = [adapters[i](sf) for i, sf in enumerate(sel)]
            
            if aggregator: emb = aggregator(sel)
            elif fusion == "concat": emb = torch.cat(sel, dim=-1)
            elif fusion == "mean": emb = torch.stack(sel, dim=0).mean(0)
            elif fusion == "sum": emb = torch.stack(sel, dim=0).sum(0)
            
            corr += (probe(emb).argmax(1) == y).sum().item()
            tot += y.shape[0]
    return corr / tot if tot > 0 else 0.0

@hydra.main(config_path="conf", config_name="config_text")
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

        dataset_path = cfg.dataset.path if cfg.dataset.path else cfg.dataset.name
        subsets = list(cfg.dataset.subsets) if cfg.dataset.subsets else [None]

        for subset in subsets:
            print(f"==========================================")
            print(f"PROCESSING SUBSET: {subset if subset else 'Full Dataset'}")
            print(f"==========================================")

            train_ds = TextDataset(dataset_path, subset, cfg.dataset.train_split, tokenizer)
            
            try:
                val_ds = TextDataset(dataset_path, subset, cfg.dataset.val_split, tokenizer)
                print(f"Loaded Validation Split: {len(val_ds)} samples")
            except Exception as e:
                print(f"Warning: Validation split loading failed ({e}). Splitting train...")
                val_ds = train_ds 
            
            if cfg.dataset.test_split:
                try:
                    test_ds = TextDataset(dataset_path, subset, cfg.dataset.test_split, tokenizer)
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
                if exp_str == "last":
                    exp_cfg.selection.mode = "last"
                    exp_cfg.topk = 1
                    exp_cfg.ablation.use_geo_loss = False
                    exp_cfg.ablation.fusion = "concat"
                elif exp_str == "last_3_mean":
                    exp_cfg.selection.mode = "last_k"
                    exp_cfg.topk = 3
                    exp_cfg.ablation.use_geo_loss = False
                    exp_cfg.ablation.fusion = "mean"
                elif exp_str == "last_3_concat":
                    exp_cfg.selection.mode = "last_k"
                    exp_cfg.topk = 3
                    exp_cfg.ablation.use_geo_loss = False
                    exp_cfg.ablation.fusion = "concat"
                elif "learnable_weight" in exp_str:
                    exp_cfg.selection.mode = "learnable_weight"
                    exp_cfg.topk = 0 
                    exp_cfg.ablation.fusion = "sum"
                    exp_cfg.ablation.use_geo_loss = True
                elif "loes" in exp_str:
                    exp_cfg.selection.mode = "loes"
                    if "_k_" in exp_str: exp_cfg.topk = int(exp_str.split("_k_")[1].split("_")[0])
                    if "no_geo" in exp_str: exp_cfg.ablation.use_geo_loss = False
                    if "mean_fusion" in exp_str: exp_cfg.ablation.fusion = "mean"
                elif "random" in exp_str:
                    exp_cfg.selection.mode = "random"
                    if "_k_" in exp_str: exp_cfg.topk = int(exp_str.split("_k_")[1].split("_")[0])

                net = BertEncoder(model_path).to(device)
                if not exp_cfg.optim.finetune:
                    for p in net.parameters(): p.requires_grad = False
                    
                n_cal = 0.2 * len(train_ds)
                embeddings, labels = collect_calibration_embeddings(net, cal_loader, n_cal, device)                #Convert embeddings to float32
                embeddings = [e.to(torch.float32) for e in embeddings]
                total_layers = len(embeddings)
                
                # --- SELECTION LOGIC ---
                if exp_cfg.selection.mode == "loes": 
                    topk = loes_select_layers(embeddings, labels, K=exp_cfg.topk)
                elif exp_cfg.selection.mode == "random": 
                    topk = random.sample(range(total_layers), exp_cfg.topk)
                elif exp_cfg.selection.mode == "last": 
                    topk = [total_layers-1]
                elif exp_cfg.selection.mode == "last_k":
                    topk = list(range(total_layers - exp_cfg.topk, total_layers))
                elif exp_cfg.selection.mode == "learnable_weight":
                    topk = list(range(total_layers))
                else: 
                    topk = list(range(total_layers))

                print(f"Selected Layers: {topk}")
                D = net.hidden_size
                proj_dim = exp_cfg.model.proj_dim
                
                adapters = None
                if not exp_cfg.ablation.no_adapters:
                    adapters = nn.ModuleList([nn.Sequential(nn.LayerNorm(D), nn.Linear(D, proj_dim), nn.GELU()) for _ in topk]).to(device)
                
                aggregator = None
                if exp_cfg.selection.mode == "learnable_weight":
                    aggregator = LearnableWeighting(len(topk)).to(device)
                    fused_dim = proj_dim
                elif exp_cfg.ablation.fusion == "concat": 
                    fused_dim = proj_dim * len(topk)
                else: 
                    fused_dim = proj_dim

                num_classes = len(set(train_ds.ds[train_ds.label_key]))
                probe = nn.Sequential(nn.LayerNorm(fused_dim), nn.Dropout(0.2), nn.Linear(fused_dim, num_classes)).to(device)
                
                # --- OPTIMIZED OPTIMIZER & SCHEDULER ---
                params = [{"params": probe.parameters(), "lr": exp_cfg.optim.lr_probe}]
                if adapters: params.append({"params": adapters.parameters(), "lr": exp_cfg.optim.lr_probe})
                if aggregator: params.append({"params": aggregator.parameters(), "lr": 1e-3})
                
                # Only add backbone params if finetuning is enabled
                if exp_cfg.optim.finetune: 
                    params.append({"params": net.parameters(), "lr": exp_cfg.optim.lr_backbone})
                
                opt = torch.optim.AdamW(params, weight_decay=exp_cfg.optim.weight_decay)
                
                # Use Hugging Face Scheduler with Warmup for stability
                total_steps = len(train_loader) * exp_cfg.training.epochs
                warmup_steps = int(total_steps * exp_cfg.training.warmup_ratio)
                
                sched = get_cosine_schedule_with_warmup(
                    opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps
                )

                geo_loss = GeometricLoss(weight=0.1 if exp_cfg.ablation.use_geo_loss else 0.0)

                ds_name = f"{exp_cfg.dataset.name}/{subset}" if subset else exp_cfg.dataset.name
                run_name = f"{exp_str}_{ds_name.replace('/', '_')}"
                
                # --- BEST MODEL SAVING ---
                best_val_acc = 0.0
                ckpt_path = f"best_model_{run_name}.pt"
                
                wandb.init(project=exp_cfg.wandb.project, name=run_name, config=OmegaConf.to_container(exp_cfg), reinit=True)
                
                for ep in range(exp_cfg.training.epochs):
                    net.train() if exp_cfg.optim.finetune else net.eval()
                    probe.train()
                    for batch in tqdm.tqdm(train_loader, desc=f"Ep {ep} - {subset}"):
                        opt.zero_grad()
                        ids, mask, y = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
                        feats = net(ids, mask, return_layers=True)
                        sel_feats = [feats[i] for i in topk]
                        if adapters: sel_feats = [adapters[i](f) for i, f in enumerate(sel_feats)]
                        
                        if aggregator: emb = aggregator(sel_feats)
                        elif exp_cfg.ablation.fusion == "concat": emb = torch.cat(sel_feats, dim=-1)
                        elif exp_cfg.ablation.fusion == "mean": emb = torch.stack(sel_feats, dim=0).mean(0)
                        elif exp_cfg.ablation.fusion == "sum": emb = torch.stack(sel_feats, dim=0).sum(0)
                        
                        loss = F.cross_entropy(probe(emb), y) + geo_loss(emb, y)
                        loss.backward()
                        
                        # Gradient Clipping for Stability
                        torch.nn.utils.clip_grad_norm_(
                            net.parameters() if exp_cfg.optim.finetune else probe.parameters(), 
                            max_norm=1.0
                        )
                        
                        opt.step()
                        sched.step()
                        wandb.log({"train_loss": loss.item()})
                    
                    # Validation
                    val_acc = run_eval(val_loader, net, probe, adapters, aggregator, topk, exp_cfg.ablation.fusion, device)
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

                test_acc = run_eval(test_loader, net, probe, adapters, aggregator, topk, exp_cfg.ablation.fusion, device)
                wandb.log({"test_acc": test_acc})
                print(f"Final Test Acc (Best Model): {test_acc:.4f}")

                with open(exp_cfg.logging.results_csv, "a+", newline='') as f:
                    f.seek(0, os.SEEK_END)
                    w = csv.DictWriter(f, fieldnames=["run", "mode", "k", "fusion", "geo", "test_acc", "dataset", "selected_layers","subset", "model"], extrasaction='ignore')
                    if f.tell() == 0: w.writeheader()
                    w.writerow({
                        "run": run_name, "mode": exp_cfg.selection.mode, "k": exp_cfg.topk,
                        "fusion": exp_cfg.ablation.fusion, "geo": exp_cfg.ablation.use_geo_loss,
                        "test_acc": test_acc, "dataset": ds_name, "selected_layers": topk, "subset": subset, "model": model_path
                    })
                
                wandb.finish()

if __name__ == "__main__": 
    main()