# full_ablation_loes.py
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
from itertools import product
from copy import deepcopy
import math
import time

# ---------------------------
# NOTES / Expected config keys (in conf/config_text.yaml)
# ---------------------------
# cfg.model.names: list of model names (e.g., ["bert-base-uncased"])
# cfg.dataset.name/path, cfg.dataset.subsets, cfg.dataset.train_split, val_split, test_split
# cfg.selection.alpha_list, cfg.selection.gamma_list, cfg.selection.eta_list (lists for sweep)
# cfg.grid.k_list (list of K values), cfg.grid.n_cal_list (calibration sizes)
# cfg.grid.fusion_list (["concat","mean","sum","learnable"])
# cfg.grid.selection_modes (["loes","residual_only","last","last_k","random","learnable_weight"])
# cfg.exps (optional short experiment strings) — grid used if cfg.grid.enabled = True
# cfg.seeds: list of ints
# cfg.optim.finetune: bool
# cfg.training.* (batches, epochs, warmup_ratio)
# cfg.logging.results_csv, cfg.wandb.project
# ---------------------------

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
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            try:
                self.ds.save_to_disk(save_path)
            except Exception:
                pass

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
    # X: N x d, Y: N x C
    Xc, Yc = X - X.mean(0, keepdim=True), Y - Y.mean(0, keepdim=True)
    d = X.shape[1]
    A = Xc.t() @ Xc + reg * torch.eye(d, device=X.device)
    W = torch.linalg.solve(A, Xc.t() @ Yc)
    b = (Y.mean(0, keepdim=True) - X.mean(0, keepdim=True) @ W).squeeze(0)
    return W, b


def collect_calibration_embeddings(net, loader, n_cal, device="cuda"):
    net.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids, mask, y = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            f = net(ids, mask, return_layers=True)
            f = [t.detach().cpu() for t in f]
            if not embeddings:
                embeddings = [[] for _ in f]
            for i, t in enumerate(f):
                embeddings[i].append(t)
            labels.append(y.cpu())
            if sum(len(b) for b in labels) >= n_cal:
                break
    embeddings = [torch.cat(e)[:n_cal].to(device) for e in embeddings]
    labels = torch.cat(labels)[:n_cal].long().to(device)
    return embeddings, labels


def loes_select_layers(embeddings, labels, K, reg=1e-3, alpha=1.0, gamma=0.5, eta=0.1, sample_tri=200):
    """
    embeddings: list of tensors [N x d_l] for each layer
    returns selected layer indices (list)
    """
    Y = F.one_hot(labels, int(labels.max()) + 1).float()
    # pick best single layer by (res_loss + alpha*(1-iso))
    best = (float("inf"), -1, None)
    for i, X in enumerate(embeddings):
        W, b = closed_form_ridge(X, Y, reg)
        loss = ((X @ W + b - Y) ** 2).mean().item()
        iso = compute_isotropy(X)
        score = loss + alpha * (1 - iso)
        if score < best[0]:
            best = (score, i, (W, b))
    selected = [best[1]]
    X_S = embeddings[best[1]].clone()
    y_hat = embeddings[best[1]] @ best[2][0] + best[2][1]
    residual = Y - y_hat

    # iterative greedy selection
    while len(selected) < K:
        cand_best = (float("inf"), None, None)
        for i, X in enumerate(embeddings):
            if i in selected: continue
            Xc, XS_c = X - X.mean(0, keepdim=True), X_S - X_S.mean(0, keepdim=True)
            # orth solve to remove projection onto X_S
            try:
                I = torch.eye(XS_c.shape[1], device=XS_c.device, dtype=XS_c.dtype)
                B_orth = torch.linalg.solve(XS_c.t() @ XS_c + 1e-6 * I, XS_c.t() @ Xc)
                X_tilde = Xc - XS_c @ B_orth + X.mean(0, keepdim=True)
            except RuntimeError:
                X_tilde = X  # fallback
            W, b = closed_form_ridge(X_tilde, residual, reg)
            res_loss = ((X_tilde @ W + b - residual) ** 2).mean().item()
            iso = compute_isotropy(X)
            # redundancy proxy: max cosine with already selected layers (use flattened)
            try:
                red = max([(torch.norm(X.t() @ embeddings[j]) / (torch.norm(X) * torch.norm(embeddings[j]))).item() for j in selected])
            except Exception:
                red = 0.0
            # triangle-area proxy
            classes = torch.unique(labels)
            cents = None
            if len(classes) >= 3:
                cents = torch.stack([X_tilde[labels == c].mean(0) for c in classes])
            tri = 0.0
            if cents is not None and cents.shape[0] >= 3:
                idx = torch.randint(0, len(cents), (min(sample_tri, max(3, len(cents))), 3), device=cents.device)
                a, b_pt, c_pt = cents[idx[:, 0]], cents[idx[:, 1]], cents[idx[:, 2]]
                ab, ac = a - b_pt, a - c_pt
                areas = 0.5 * torch.sqrt((ab.pow(2).sum(1) * ac.pow(2).sum(1) - (ab * ac).sum(1).pow(2)).clamp(min=0))
                tri = areas.mean().item()
            score = res_loss + alpha * (1 - iso) + gamma * red - eta * tri
            if score < cand_best[0]:
                cand_best = (score, i, (W, b, X_tilde))
        if cand_best[1] is None:
            break
        idx = cand_best[1]
        # update y_hat with full embeddings (not X_tilde)
        W_f, b_f = closed_form_ridge(embeddings[idx], residual + y_hat, reg)
        y_hat = y_hat + embeddings[idx] @ W_f + b_f
        residual = Y - y_hat
        # append selected features to X_S (concatenation along feature dim)
        X_S = torch.cat([X_S, embeddings[idx]], dim=1)
        selected.append(idx)
    return selected


class GeometricLoss(nn.Module):
    def __init__(self, weight=0.1, sample_tri=3):
        super().__init__()
        self.weight = weight
        self.sample_tri = sample_tri

    def forward(self, feats, labels):
        if self.weight <= 0:
            return torch.tensor(0.0, device=feats.device)
        classes = torch.unique(labels)
        if len(classes) < 3:
            return torch.tensor(0.0, device=feats.device)
        centroids = torch.stack([feats[labels == c].mean(0) for c in classes])
        if centroids.shape[0] < 3:
            return torch.tensor(0.0, device=feats.device)
        # pick three random centroids (or first three)
        if centroids.shape[0] >= 3:
            idx = torch.randperm(len(centroids), device=centroids.device)[:3]
            ab = centroids[idx[0]] - centroids[idx[1]]
            ac = centroids[idx[0]] - centroids[idx[2]]
            area = 0.5 * torch.sqrt((ab.pow(2).sum() * ac.pow(2).sum() - (ab * ac).sum().pow(2)).clamp(min=1e-6))
        else:
            area = torch.tensor(1.0, device=feats.device)
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
                sel = [adapters[i](sf) for i, sf in enumerate(sel)]
            if aggregator:
                emb = aggregator(sel)
            elif fusion == "concat":
                emb = torch.cat(sel, dim=-1)
            elif fusion == "mean":
                emb = torch.stack(sel, dim=0).mean(0)
            elif fusion == "sum":
                emb = torch.stack(sel, dim=0).sum(0)
            else:
                emb = torch.cat(sel, dim=-1)
            corr += (probe(emb).argmax(1) == y).sum().item()
            tot += y.shape[0]
    return corr / tot if tot > 0 else 0.0


def build_grid(cfg):
    # Build lists (fall back to singletons if not provided)
    alpha_list = getattr(cfg.selection, "alpha_list", [getattr(cfg.selection, "alpha", 1.0)])
    gamma_list = getattr(cfg.selection, "gamma_list", [getattr(cfg.selection, "gamma", 0.5)])
    eta_list = getattr(cfg.selection, "eta_list", [getattr(cfg.selection, "eta", 0.1)])
    k_list = getattr(cfg.grid, "k_list", [getattr(cfg, "topk", 3)])
    n_cal_list = getattr(cfg.grid, "n_cal_list", [getattr(cfg.calibration, "n_cal", 256)])
    fusion_list = getattr(cfg.grid, "fusion_list", [cfg.ablation.fusion])
    modes = getattr(cfg.grid, "selection_modes", [cfg.selection.mode])
    seeds = getattr(cfg, "seeds", [cfg.seed])
    geo_flags = getattr(cfg.grid, "geo_flags", [cfg.ablation.use_geo_loss])
    return list(product(modes, alpha_list, gamma_list, eta_list, k_list, n_cal_list, fusion_list, geo_flags, seeds))


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Not forcing full determinism to preserve performance, but seed is set.


def run_single_experiment(exp_cfg, base_cfg, model_path, tokenizer, train_ds, cal_loader, train_loader, val_loader, test_loader, subset, ds_name, run_id):
    """
    exp_cfg: a DictConfig copy with selection.alpha/gamma/eta/topk etc set for this run
    base_cfg: original cfg (for bookkeeping)
    run_id: string unique id
    """
    device = base_cfg.device
    # build net
    net = BertEncoder(model_path).to(device)
    if not exp_cfg.optim.finetune:
        for p in net.parameters():
            p.requires_grad = False

    # calibration embeddings
    n_cal = 0.2 * len(train_ds)
    embeddings, labels = collect_calibration_embeddings(net, cal_loader, n_cal, device)
    total_layers = len(embeddings)

    # selection logic
    if exp_cfg.selection.mode == "loes":
        topk = loes_select_layers(embeddings, labels, K=exp_cfg.topk, reg=exp_cfg.selection.reg, alpha=exp_cfg.selection.alpha, gamma=exp_cfg.selection.gamma, eta=exp_cfg.selection.eta)
    elif exp_cfg.selection.mode == "residual_only":
        topk = loes_select_layers(embeddings, labels, K=exp_cfg.topk, reg=exp_cfg.selection.reg, alpha=0.0, gamma=0.0, eta=0.0)
    elif exp_cfg.selection.mode == "random":
        topk = random.sample(range(total_layers), exp_cfg.topk)
    elif exp_cfg.selection.mode == "last":
        topk = [total_layers - 1]
    elif exp_cfg.selection.mode == "last_k":
        topk = list(range(total_layers - exp_cfg.topk, total_layers))
    elif exp_cfg.selection.mode == "learnable_weight":
        topk = list(range(total_layers))
    else:
        topk = list(range(total_layers))

    D = net.hidden_size
    proj_dim = exp_cfg.model.proj_dim

    adapters = None
    if not exp_cfg.ablation.no_adapters:
        adapters = nn.ModuleList([nn.Sequential(nn.LayerNorm(D), nn.Linear(D, proj_dim), nn.GELU()) for _ in topk]).to(device)

    aggregator = None
    fused_dim = proj_dim
    if exp_cfg.selection.mode == "learnable_weight" :
        aggregator = LearnableWeighting(len(topk)).to(device)
        fused_dim = proj_dim
    elif exp_cfg.ablation.fusion == "concat":
        fused_dim = proj_dim * len(topk)
    else:
        fused_dim = proj_dim

    num_classes = len(set(train_ds.ds[train_ds.label_key]))
    probe = nn.Sequential(nn.LayerNorm(fused_dim), nn.Dropout(0.2), nn.Linear(fused_dim, num_classes)).to(device)

    # optimizer
    params = [{"params": probe.parameters(), "lr": exp_cfg.optim.lr_probe}]
    if adapters:
        params.append({"params": adapters.parameters(), "lr": exp_cfg.optim.lr_probe})
    if aggregator:
        params.append({"params": aggregator.parameters(), "lr": exp_cfg.optim.lr_aggregator if hasattr(exp_cfg.optim, "lr_aggregator") else 1e-3})
    if exp_cfg.optim.finetune:
        params.append({"params": net.parameters(), "lr": exp_cfg.optim.lr_backbone})

    opt = torch.optim.AdamW(params, weight_decay=exp_cfg.optim.weight_decay)
    total_steps = max(1, len(train_loader) * exp_cfg.training.epochs)
    warmup_steps = int(total_steps * exp_cfg.training.warmup_ratio)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    geo_loss = GeometricLoss(weight=exp_cfg.ablation.geo_weight if exp_cfg.ablation.use_geo_loss else 0.0)

    run_name = f"{run_id}_{model_path.split('/')[-1]}_{ds_name.replace('/', '_')}"
    wandb.init(project=exp_cfg.wandb.project, name=run_name, config=OmegaConf.to_container(exp_cfg), reinit=True)

    best_val_acc = 0.0
    ckpt_path = f"best_model_{run_name}.pt"

    for ep in range(exp_cfg.training.epochs):
        net.train() if exp_cfg.optim.finetune else net.eval()
        probe.train()
        for batch in tqdm.tqdm(train_loader, desc=f"Ep {ep} - {subset}"):
            opt.zero_grad()
            ids, mask, y = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            feats = net(ids, mask, return_layers=True)
            sel_feats = [feats[i] for i in topk]
            if adapters:
                sel_feats = [adapters[i](f) for i, f in enumerate(sel_feats)]
            if aggregator:
                emb = aggregator(sel_feats)
            elif exp_cfg.ablation.fusion == "concat":
                emb = torch.cat(sel_feats, dim=-1)
            elif exp_cfg.ablation.fusion == "mean":
                emb = torch.stack(sel_feats, dim=0).mean(0)
            elif exp_cfg.ablation.fusion == "sum":
                emb = torch.stack(sel_feats, dim=0).sum(0)
            else:
                emb = torch.cat(sel_feats, dim=-1)

            loss = F.cross_entropy(probe(emb), y) + geo_loss(emb, y)
            loss.backward()
            # gradient clipping
            params_for_clip = net.parameters() if exp_cfg.optim.finetune else probe.parameters()
            torch.nn.utils.clip_grad_norm_(params_for_clip, max_norm=1.0)
            opt.step()
            try:
                sched.step()
            except Exception:
                pass
            wandb.log({"train_loss": loss.item(), "epoch": ep})

        # validation
        val_acc = run_eval(val_loader, net, probe, adapters, aggregator, topk, exp_cfg.ablation.fusion, base_cfg.device)
        wandb.log({"val_acc": val_acc})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state = {
                'probe': probe.state_dict(),
                'adapters': adapters.state_dict() if adapters else None,
                'aggregator': aggregator.state_dict() if aggregator else None,
                'net': net.state_dict() if exp_cfg.optim.finetune else None
            }
            torch.save(state, ckpt_path)

    # load best
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=base_cfg.device)
        probe.load_state_dict(checkpoint['probe'])
        if adapters and checkpoint.get('adapters'):
            adapters.load_state_dict(checkpoint['adapters'])
        if aggregator and checkpoint.get('aggregator'):
            aggregator.load_state_dict(checkpoint['aggregator'])
        if exp_cfg.optim.finetune and checkpoint.get('net'):
            net.load_state_dict(checkpoint['net'])

    test_acc = run_eval(test_loader, net, probe, adapters, aggregator, topk, exp_cfg.ablation.fusion, base_cfg.device)
    wandb.log({"test_acc": test_acc})
    wandb.finish()

    result_row = {
        "run": run_name,
        "selection_mode": exp_cfg.selection.mode,
        "alpha": exp_cfg.selection.alpha,
        "gamma": exp_cfg.selection.gamma,
        "eta": exp_cfg.selection.eta,
        "k": exp_cfg.topk,
        "n_cal": exp_cfg.calibration.n_cal,
        "fusion": exp_cfg.ablation.fusion,
        "geo": exp_cfg.ablation.use_geo_loss,
        "geo_weight": exp_cfg.ablation.geo_weight,
        "test_acc": float(test_acc),
        "val_acc": float(best_val_acc),
        "dataset": ds_name,
        "selected_layers": topk,
        "model": model_path,
        "seed": exp_cfg.seed,
        "time": time.time()
    }
    # append to csv
    csv_path = base_cfg.logging.results_csv
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a+", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result_row.keys()), extrasaction='ignore')
        if f.tell() == 0:
            w.writeheader()
        w.writerow(result_row)
    return result_row


@hydra.main(config_path="conf", config_name="config_text_sweep")
def main(cfg: DictConfig):
    base_cfg = cfg
    device = cfg.device
    set_seed(cfg.seed)

    # build grid
    if getattr(cfg.grid, "enabled", False):
        grid = build_grid(cfg)
    else:
        # make single config from cfg.exps or default
        modes = getattr(cfg.grid, "selection_modes", [cfg.selection.mode])
        alpha_list = getattr(cfg.selection, "alpha_list", [cfg.selection.alpha])
        gamma_list = getattr(cfg.selection, "gamma_list", [cfg.selection.gamma])
        eta_list = getattr(cfg.selection, "eta_list", [cfg.selection.eta])
        k_list = getattr(cfg.grid, "k_list", [cfg.topk])
        n_cal_list = getattr(cfg.grid, "n_cal_list", [cfg.calibration.n_cal])
        fusion_list = getattr(cfg.grid, "fusion_list", [cfg.ablation.fusion])
        geo_flags = getattr(cfg.grid, "geo_flags", [cfg.ablation.use_geo_loss])
        seeds = getattr(cfg, "seeds", [cfg.seed])
        grid = list(product(modes, alpha_list, gamma_list, eta_list, k_list, n_cal_list, fusion_list, geo_flags, seeds))

    # iterate models & datasets
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
            print("==========================================")
            print(f"PROCESSING SUBSET: {subset if subset else 'Full Dataset'}")
            print("==========================================")

            train_ds = TextDataset(dataset_path, subset, cfg.dataset.train_split, tokenizer)
            try:
                val_ds = TextDataset(dataset_path, subset, cfg.dataset.val_split, tokenizer)
            except Exception as e:
                print(f"Warning: Validation split loading failed ({e}). Splitting train...")
                val_ds = train_ds

            if cfg.dataset.test_split:
                try:
                    test_ds = TextDataset(dataset_path, subset, cfg.dataset.test_split, tokenizer)
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

            ds_name = f"{cfg.dataset.name}/{subset}" if subset else cfg.dataset.name

            # loop grid
            for (mode, alpha, gamma, eta, k, n_cal, fusion, geo_flag, seed) in grid:
                # build a run-specific config (deepcopy)
                exp_cfg = deepcopy(cfg)
                exp_cfg.selection.mode = mode
                exp_cfg.selection.alpha = float(alpha)
                exp_cfg.selection.gamma = float(gamma)
                exp_cfg.selection.eta = float(eta)
                exp_cfg.topk = int(k)
                exp_cfg.calibration.n_cal = int(n_cal)
                exp_cfg.ablation.fusion = fusion
                exp_cfg.ablation.use_geo_loss = bool(geo_flag)
                exp_cfg.seed = int(seed)
                exp_cfg.selection.reg = float(getattr(cfg.selection, "reg", 1e-3))
                exp_cfg.ablation.geo_weight = float(getattr(cfg.ablation, "geo_weight", 0.1))
                # for aggregator learning rate (optional)
                exp_cfg.optim.lr_aggregator = float(getattr(cfg.optim, "lr_aggregator", 1e-3))
                # exp_cfg.grid_fusion = fusion

                # set seeds
                set_seed(exp_cfg.seed)

                # build a readable run id
                run_id = f"{mode}_a{exp_cfg.selection.alpha}_g{exp_cfg.selection.gamma}_e{exp_cfg.selection.eta}_k{k}_ncal{n_cal}_fusion{fusion}_geo{int(geo_flag)}_s{exp_cfg.seed}"

                print(f"Starting run: {run_id} on dataset {ds_name}")

                try:
                    result = run_single_experiment(exp_cfg, base_cfg, model_path, tokenizer, train_ds, cal_loader, train_loader, val_loader, test_loader, subset, ds_name, run_id)
                    print("Result:", result)
                except Exception as e:
                    print(f"Run {run_id} failed with exception: {e}")
                    # log failure row
                    fail_row = {"run": run_id, "error": str(e), "dataset": ds_name, "model": model_path}
                    with open(cfg.logging.results_csv, "a+", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(fail_row.keys()), extrasaction='ignore')
                        if f.tell() == 0:
                            w.writeheader()
                        w.writerow(fail_row)
                    continue

if __name__ == "__main__":
    main()
