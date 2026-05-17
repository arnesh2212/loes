# %%
import random
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2
import timm
import wandb
import hydra
import tqdm
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import CLIPModel, AutoProcessor, get_cosine_schedule_with_warmup


# %%
device='cpu'
if torch.cuda.is_available():
    device='cuda'
device

# %%
class HFCLIPImageTextEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()

        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

        self.img_hidden_dim = self.model.vision_model.config.hidden_size
        self.txt_hidden_dim = self.model.text_model.config.hidden_size
        self.n_layers = self.model.vision_model.config.num_hidden_layers

        self.pooling = "mean"

        print(
            f"Loaded CLIP: img_dim={self.img_hidden_dim}, "
            f"text_dim={self.txt_hidden_dim}, layers={self.n_layers}"
        )

    def _mean_pool(self, hidden_state, mask=None):
        if mask is None:
            return hidden_state.mean(dim=1)

        mask = mask.unsqueeze(-1).to(hidden_state.dtype)
        return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

    def forward(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        return_layers=False
):  
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
    
        img_hidden_states = outputs.vision_model_output.hidden_states
        txt_hidden_states = outputs.text_model_output.hidden_states
    
        if return_layers:
            feats = []
            for img_hs, txt_hs in zip(img_hidden_states, txt_hidden_states):
                img_feat = self._mean_pool(img_hs)
                txt_feat = self._mean_pool(txt_hs, attention_mask)
                feats.append(torch.cat([img_feat, txt_feat], dim=-1))
            return feats
    
        img_feat = self._mean_pool(img_hidden_states[-1])
        txt_feat = self._mean_pool(txt_hidden_states[-1], attention_mask)
        return torch.cat([img_feat, txt_feat], dim=-1)
    

# %% [markdown]
# ##### Dataloader

# %%
# Download data
#import kagglehub
#path = kagglehub.dataset_download("parthplc/facebook-hateful-meme-dataset")
#path+='/data'
#print("Path to dataset files:", path)

# %%
#* DATALOADER FOR AMAZON PRODUCTS
import os
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image

class CLIPDataset_Amzprod(Dataset):
    def __init__(
        self,
        root_dir,
        csv_file,
        tokenizer,
        image_transform=None,
        max_length=77
    ):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length

        self.data = []
        with open(os.path.join(root_dir, csv_file), newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row)

        if self.image_transform is not None:
            dummy = Image.new("RGB", (224, 224))
            self.zero_image = torch.zeros_like(self.image_transform(dummy))
        else:
            self.zero_image = torch.zeros(3, 224, 224)

        #self.image_dir = os.path.join(
        #    root_dir, os.path.splitext(csv_file)[0]
        #)
        self.image_dir= root_dir+'/'+csv_file.split('_')[0]
        print(self.image_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        id = item["asin"]
        title = item["title"]
        price = float(item["price"])   #* regression

        img_path = os.path.join(self.image_dir, f"{id}.jpg")

        try:
            image = Image.open(img_path).convert("RGB")
            if self.image_transform:
                image = self.image_transform(image)
        except Exception:
            image = self.zero_image

        tokens = self.tokenizer(
            title,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        label = torch.tensor(price, dtype=torch.float)     #* regression
        #label = torch.tensor(item["6_way_label"], dtype=torch.long)    #* classification

        return {
            "pixel_values": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label
        }



# %%
from torchvision.transforms import v2
from transformers import CLIPTokenizer
from torch.utils.data import DataLoader
import torch

def get_dataloader(
    root_path,
    filename: str,
    batch_size=32,
    img_size=224,
    model_name="openai/clip-vit-base-patch32"
):
    # ---- CLIP-specific normalization ----
    transform = v2.Compose([
        v2.Resize(img_size),
        v2.CenterCrop(img_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    tokenizer = CLIPTokenizer.from_pretrained(model_name)

    dset = CLIPDataset_Amzprod(
        root_dir=root_path,
        csv_file=filename,
        tokenizer=tokenizer,
        image_transform=transform
    )

    dloader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    return dloader, dset




# %%


# %%
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
        for batch in loader:
            pix_val,ids, mask, y = batch["pixel_values"].to(device),batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            f = [t.cpu() for t in net(pix_val, ids, mask, return_layers=True)]
            if not embeddings: embeddings = [[] for _ in f]
            for i, t in enumerate(f): embeddings[i].append(t)
            labels.append(y)
            if sum(len(b) for b in labels) >= n_cal: break
    return [torch.cat(e)[:n_cal].to(device) for e in embeddings], torch.cat(labels)[:n_cal].float().to(device)   #* Use for regression
    #!return [torch.cat(e)[:n_cal].to(device) for e in embeddings], torch.cat(labels)[:n_cal].long().to(device)




# %%
def loes_select_layers(embeddings:list[torch.Tensor], labels:torch.Tensor, K:int, reg:float=1e-3, alpha:float=1.0, gamma:float=0.5, eta:float=0.1,task:str='classification'):
    
    if task=='classification':
        Y = F.one_hot(labels, int(labels.max())+1).float()
    else:   # regression
        Y = labels
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
            B_orth = torch.linalg.solve(XS_c.t() @ XS_c + 1e-6 * torch.eye(XS_c.shape[1],device=X.device), XS_c.t() @ Xc)
            X_tilde = Xc - XS_c @ B_orth + X.mean(0, keepdim=True)
            W, b = closed_form_ridge(X_tilde, residual, reg)
            res_loss = ((X_tilde @ W + b - residual)**2).mean().item()
            iso = compute_isotropy(X)
            red = max([ (torch.norm(X.t()@embeddings[j])/(torch.norm(X)*torch.norm(embeddings[j]))).item() for j in selected])
            
            tri = 0.0   # regression
            if task=='classification':
                classes = torch.unique(labels)
                cents = torch.stack([X_tilde[labels==c].mean(0) for c in classes]) if len(classes)>=3 else None

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
        #W_f, b_f = closed_form_ridge(embeddings[idx], residual, reg)
        y_hat += embeddings[idx] @ W_f + b_f
        y_hat += X_tilde @ W + b
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

# %% [markdown]
# #### Training/Eval

# %%
def run_eval_class(loader, net, probe, adapters, aggregator, topk, fusion, device):
    net.eval(); probe.eval()
    corr, tot = 0, 0
    with torch.no_grad():
        for batch in loader:
            pix_val,ids, mask, y = batch["pixel_values"].to(device),batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            f = net(pix_val, ids, mask, return_layers=True)
            sel = [f[i] for i in topk]
            if adapters: sel = [adapters[i](sf) for i, sf in enumerate(sel)]
            
            if aggregator: emb = aggregator(sel)
            elif fusion == "concat": emb = torch.cat(sel, dim=-1)
            elif fusion == "mean": emb = torch.stack(sel, dim=0).mean(0)
            elif fusion == "sum": emb = torch.stack(sel, dim=0).sum(0)
            
            corr += (probe(emb).argmax(1) == y).sum().item()
            tot += y.shape[0]
    return corr / tot if tot > 0 else 0.0


def run_eval_reg(loader, net, probe, adapters, aggregator, topk, fusion, device):
    net.eval(); probe.eval()
    total_squared_error = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            pix_val,ids, mask, y = batch["pixel_values"].to(device),batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            f = net(pix_val, ids, mask, return_layers=True)
            sel = [f[i] for i in topk]
            if adapters: sel = [adapters[i](sf) for i, sf in enumerate(sel)]
            
            if aggregator: emb = aggregator(sel)
            elif fusion == "concat": emb = torch.cat(sel, dim=-1)
            elif fusion == "mean": emb = torch.stack(sel, dim=0).mean(0)
            elif fusion == "sum": emb = torch.stack(sel, dim=0).sum(0)
            

            # Prediction
            preds = probe(emb).view_as(y) # Matches shape of y exactly
            
            # Calculate SUM of squared errors for this batch
            batch_mse = F.mse_loss(preds, y, reduction='sum').item()
            
            total_squared_error += batch_mse
            total_samples += y.size(0)

    # Final RMS Calculation: Sqrt of (Total SSE / Total N)
    final_rmse = torch.sqrt(torch.tensor(total_squared_error / total_samples))
    return final_rmse.item()



def run_eval(loader, net, probe, adapters, aggregator, topk, fusion, device, task:str='classification'):
    if task=='classification':
        return run_eval_class(loader, net, probe, adapters, aggregator, topk, fusion, device)
    else:   # regression
        return run_eval_reg(loader, net, probe, adapters, aggregator, topk, fusion, device)


# %%
#%tb
@hydra.main(config_path="conf", config_name="img_text_config")
def main(cfg: DictConfig):
    device = cfg.device
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    for model_path in cfg.model.names:
        #tokenizer = AutoTokenizer.from_pretrained(model_path)

        #def collate_fn(batch):
        #    texts, labels = zip(*batch)
        #    toks = tokenizer(list(texts), padding=True, truncation=True, max_length=cfg.model.max_len, return_tensors="pt")
        #    toks['labels'] = torch.tensor(labels)
        #    return toks

        #dataset_path = cfg.dataset.path if cfg.dataset.path else cfg.dataset.name
        #subsets = list(cfg.dataset.subsets) if cfg.dataset.subsets else [None]

        #for subset in subsets:
        #    print(f"==========================================")
        #    print(f"PROCESSING SUBSET: {subset if subset else 'Full Dataset'}")
        #    print(f"==========================================")
#
        #    train_ds = TextDataset(dataset_path, subset, cfg.dataset.train_split, tokenizer)
        #    
        #    try:
        #        val_ds = TextDataset(dataset_path, subset, cfg.dataset.val_split, tokenizer)
        #        print(f"Loaded Validation Split: {len(val_ds)} samples")
        #    except Exception as e:
        #        print(f"Warning: Validation split loading failed ({e}). Splitting train...")
        #        val_ds = train_ds 
        #    
        #    if cfg.dataset.test_split:
        #        try:
        #            test_ds = TextDataset(dataset_path, subset, cfg.dataset.test_split, tokenizer)
        #            print(f"Loaded Test Split: {len(test_ds)} samples")
        #        except Exception as e:
        #            print(f"Test split loading failed ({e}). Using Validation split as Test.")
        #            test_ds = val_ds
        #    else:
        #        print("Test split is null in config. Using Validation split as Test.")
        #        test_ds = val_ds
#
        #    train_loader = DataLoader(train_ds, cfg.training.bs, shuffle=True, collate_fn=collate_fn, num_workers=4)
        #    cal_loader = DataLoader(train_ds, cfg.calibration.cal_bs, shuffle=True, collate_fn=collate_fn)
        #    val_loader = DataLoader(val_ds, cfg.training.test_bs, collate_fn=collate_fn, num_workers=4)
        #    test_loader = DataLoader(test_ds, cfg.training.test_bs, collate_fn=collate_fn, num_workers=4)

            
            #root="/home/aniket/.cache/kagglehub/datasets/parthplc/facebook-hateful-meme-dataset/versions/1/data"    #! replace with actual dir
            root= cfg.dataset.path
            train_loader, train_ds = get_dataloader(root_path=root,filename="train.csv",batch_size=cfg.training.bs,img_size=cfg.model.img_size)
            _, cal_ds = get_dataloader(root_path=root,filename="train.csv",batch_size=cfg.calibration.cal_bs,img_size=cfg.model.img_size)
            test_loader, _ = get_dataloader(root_path=root,filename="test.csv",batch_size=cfg.training.test_bs,img_size=cfg.model.img_size)
            val_loader, _ = get_dataloader(root_path=root,filename="val.csv",batch_size=cfg.training.test_bs,img_size=cfg.model.img_size)

            
            experiments = cfg.exps if cfg.exps else ["default"]
            
            for exp_str in experiments:
                print(f"--- Starting Experiment: {exp_str} ---")
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

                #net = BertEncoder(model_path).to(device)
                net = HFCLIPImageTextEncoder().to(device)
                if not exp_cfg.optim.finetune:
                    for p in net.parameters(): p.requires_grad = False
                    
                embeddings, labels = collect_calibration_embeddings(net, cal_ds, exp_cfg.calibration.n_cal,cfg.calibration.cal_bs ,device)
                #embeddings, labels = collect_calibration_embeddings(net, cal_ds, 1700, 32,device)
                total_layers = len(embeddings)
                
                # --- SELECTION LOGIC ---
                if exp_cfg.selection.mode == "loes": 
                    topk = loes_select_layers(embeddings, labels, K=exp_cfg.topk,task=exp_cfg.task)
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
                continue
                import sys
                sys.exit()
                #D = net.hidden_size
                D = 1280    #! change this acc to model (1280 for clip img+text)
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

                
                #! Major changes made here for updating classification to regression
                num_classes = 1 #* Regression
                if exp_cfg.task=='classification':
                    #num_classes = len(set(train_ds.ds[train_ds.label_key]))
                    num_classes = 6     #! hardcoded
                
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

                
                #geo_loss = 0 #* set geo loss 0 if backbone is freezed
                #if exp_cfg.optim.finetune:  #* trainable backbone
                geo_loss = GeometricLoss(weight=0.1 if exp_cfg.ablation.use_geo_loss else 0.0)

                ds_name = f"{exp_cfg.dataset.name}"
                run_name = f"{exp_str}_{ds_name.replace('/', '_')}"
                
                # --- BEST MODEL SAVING ---
                best_val_metric = float('inf')
                ckpt_path = f"best_model_{run_name}.pt"
                
                wandb.init(project=exp_cfg.wandb.project, name=run_name, config=OmegaConf.to_container(exp_cfg), reinit=True)
                
                for ep in range(exp_cfg.training.epochs):
                    net.train() if exp_cfg.optim.finetune else net.eval()
                    probe.train()
                    for batch in tqdm.tqdm(train_loader, desc=f"Ep {ep}"):
                        opt.zero_grad()
                        pix_val, ids, mask, y = batch["pixel_values"].to(device), batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
                        feats = net(pix_val, ids, mask, return_layers=True)
                        sel_feats = [feats[i] for i in topk]
                        if adapters: sel_feats = [adapters[i](f) for i, f in enumerate(sel_feats)]
                        
                        if aggregator: emb = aggregator(sel_feats)
                        elif exp_cfg.ablation.fusion == "concat": emb = torch.cat(sel_feats, dim=-1)
                        elif exp_cfg.ablation.fusion == "mean": emb = torch.stack(sel_feats, dim=0).mean(0)
                        elif exp_cfg.ablation.fusion == "sum": emb = torch.stack(sel_feats, dim=0).sum(0)
                        
                        geo_loss_val=0
                        if exp_cfg.optim.finetune:
                            geo_loss_val=geo_loss(emb, y)
                        
                        if exp_cfg.task=='classification':
                            loss = F.cross_entropy(probe(emb), y) + geo_loss_val
                        else:   #* regression
                            loss = F.mse_loss(probe(emb).squeeze(-1), y) + geo_loss_val
                        
                        loss.backward()     #* BACKPROP !!!
                        
                        # Gradient Clipping for Stability
                        torch.nn.utils.clip_grad_norm_(
                            net.parameters() if exp_cfg.optim.finetune else probe.parameters(), 
                            max_norm=1.0
                        )
                        
                        opt.step()
                        sched.step()
                        wandb.log({"train_loss": loss.item()})
                    
                    # Validation
                    metric_name='rmse'   #* regression
                    if exp_cfg.task=='classification':
                        metric_name='acc'

                    val_metric = run_eval(val_loader, net, probe, adapters, aggregator, topk, exp_cfg.ablation.fusion, device,exp_cfg.task)
                    wandb.log({f"val_{metric_name}": val_metric})
                    print(f"Ep {ep} Val {metric_name}: {val_metric:.4f}")
                    
                    #! if val_metric > best_val_metric:
                    if (metric_name =='rmse' and val_metric < best_val_metric) or (metric_name =='acc' and val_metric > best_val_metric):    #* check if current < best for rmse and current > best for acc
                        best_val_metric = val_metric
                        print(f"New best Val {metric_name}: {val_metric:.4f}. Saving checkpoint to {ckpt_path}")
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

                test_metric = run_eval(test_loader, net, probe, adapters, aggregator, topk, exp_cfg.ablation.fusion, device, exp_cfg.task)
                wandb.log({f"test_{metric_name}": test_metric})
                print(f"Final Test {metric_name} (Best Model): {test_metric:.4f}")

                with open(exp_cfg.logging.results_csv, "a+", newline='') as f:
                    f.seek(0, os.SEEK_END)
                    w = csv.DictWriter(f, fieldnames=["run", "mode", "k", "fusion", "geo", f"test_{metric_name}", "dataset", "selected_layers","subset", "model"], extrasaction='ignore')
                    if f.tell() == 0: w.writeheader()
                    w.writerow({
                        "run": run_name, "mode": exp_cfg.selection.mode, "k": exp_cfg.topk,
                        "fusion": exp_cfg.ablation.fusion, "geo": exp_cfg.ablation.use_geo_loss,
                        f"test_{metric_name}": test_metric, "dataset": ds_name, "selected_layers": topk, "model": model_path
                    })
                
                wandb.finish()

if __name__ == "__main__": 
    main()


