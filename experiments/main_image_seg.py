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
from transformers import AutoImageProcessor, AutoModel, AutoProcessor
from transformers import BeitForImageClassification
from pathlib import Path
from PIL import Image
import numpy as np


# =============================================================================
# HuggingFace ViT Encoder for Segmentation (returns patch tokens)
# =============================================================================

class HFViTEncoderSeg(nn.Module):
    # Returns patch tokens (not CLS) for segmentation tasks
    # Each model has different structure for extracting patches
    
    def __init__(self, model_name, img_size=224):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        
        print(f"  Loading {model_name} for segmentation...")
        
        if "clipseg" in model_name.lower():
            # CLIPSeg vision model
            from transformers import CLIPSegVisionModel
            self.model = CLIPSegVisionModel.from_pretrained(model_name)
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.embed_dim = self.model.config.hidden_size
            self.n_layers = self.model.config.num_hidden_layers
            # CLIPSeg: 485 tokens = 1 CLS + 484 patches (22x22 for 352/16)
            # For 224 input, will be different - detect at runtime
            self.patch_start_idx = 1  # Skip CLS
            self.patch_end_idx = None  # Use all after CLS
            self.num_patches = None  # Detect at runtime
            print(f"  CLIPSeg model: embed_dim={self.embed_dim}, n_layers={self.n_layers}")
            
        elif "beit" in model_name.lower():
            # BEiT model
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.embed_dim = self.model.config.hidden_size
            self.n_layers = self.model.config.num_hidden_layers
            # BEiT: 197 tokens = 1 CLS + 196 patches (14x14)
            self.patch_start_idx = 1
            self.patch_end_idx = None
            self.num_patches = 196
            print(f"  BEiT model: embed_dim={self.embed_dim}, n_layers={self.n_layers}")
            
        elif "dinov2" in model_name.lower() and "dinov3" not in model_name.lower():
            # DINOv2
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.embed_dim = self.model.config.hidden_size
            self.n_layers = self.model.config.num_hidden_layers
            # DINOv2-S with patch14: 257 tokens = 1 CLS + 256 patches (16x16 for 224/14)
            self.patch_start_idx = 1
            self.patch_end_idx = None
            self.num_patches = 256  # For 224/14
            print(f"  DINOv2 model: embed_dim={self.embed_dim}, n_layers={self.n_layers}")
            
        elif "dinov3" in model_name.lower():
            # DINOv3 with register tokens
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.embed_dim = self.model.config.hidden_size
            self.n_layers = self.model.config.num_hidden_layers
            # DINOv3-S16: 201 tokens = 1 CLS + 196 patches + 4 registers
            # Patches are at indices 1:197, registers at 197:201
            self.patch_start_idx = 1
            self.patch_end_idx = -4  # Exclude last 4 register tokens
            self.num_patches = 196
            num_reg = getattr(self.model.config, 'num_register_tokens', 4)
            print(f"  DINOv3 model: embed_dim={self.embed_dim}, n_layers={self.n_layers}, registers={num_reg}")
            
        else:
            # Generic fallback
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.embed_dim = self.model.config.hidden_size
            self.n_layers = self.model.config.num_hidden_layers
            self.patch_start_idx = 1
            self.patch_end_idx = None
            self.num_patches = None
            print(f"  Generic model: embed_dim={self.embed_dim}, n_layers={self.n_layers}")
        
        print(f"  Loaded successfully: embed_dim={self.embed_dim}, n_layers={self.n_layers}")
    
    def _extract_patch_tokens(self, hidden_state):
        # hidden_state: [B, seq_len, D]
        # Returns: [B, num_patches, D]
        if self.patch_end_idx is not None:
            patches = hidden_state[:, self.patch_start_idx:self.patch_end_idx, :]
        else:
            patches = hidden_state[:, self.patch_start_idx:, :]
        return patches
    
    def forward(self, x, return_layers=False):
        # x: [B, 3, H, W] preprocessed tensor
        outputs = self.model(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        # Debug print for first forward pass
        if not hasattr(self, '_debug_printed'):
            hs = hidden_states[1]
            patches = self._extract_patch_tokens(hs)
            print(f"  Forward pass: {len(hidden_states)} hidden states")
            print(f"  Raw hidden state shape: {hs.shape}")
            print(f"  Extracted patches shape: {patches.shape}")
            self.num_patches = patches.shape[1]
            self._debug_printed = True
        
        if return_layers:
            feats = []
            for hs in hidden_states[1:]:  # Skip embedding layer
                patches = self._extract_patch_tokens(hs)
                feats.append(patches)
            return feats  # List of [B, num_patches, D]
        else:
            return self._extract_patch_tokens(hidden_states[-1])


# =============================================================================
# Segmentation Dataset
# =============================================================================

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, split, img_size=224, num_classes=None):
        self.dataset_name = dataset_name
        self.img_size = img_size
        
        print(f"  Loading dataset: {dataset_name}, split: {split}")
        self.ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
        
        # Dataset-specific configuration
        if "ADE20K" in dataset_name or "ade20k" in dataset_name.lower():
            self.img_key = "image"
            self.mask_key = "segmentations"
            self.num_classes = num_classes or 150
            self.ignore_index = 0  # Background is 0 in ADE20K
            self.is_ade20k = True
            
        elif "cityscapes" in dataset_name.lower():
            self.img_key = "image"
            self.mask_key = "semantic_segmentation"
            self.num_classes = num_classes or 19
            self.ignore_index = 255
            self.is_ade20k = False
            
        elif "COCOStuff" in dataset_name or "cocostuff" in dataset_name.lower():
            self.img_key = "image"
            self.mask_key = "mask"
            self.num_classes = num_classes or 171
            self.ignore_index = 255
            self.is_ade20k = False
            
        elif "FoodSeg" in dataset_name or "foodseg" in dataset_name.lower():
            self.img_key = "image"
            self.mask_key = "label"
            self.num_classes = num_classes or 104
            self.ignore_index = 255
            self.is_ade20k = False
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        print(f"  Dataset loaded: {len(self.ds)} samples, {self.num_classes} classes")
        
        # Image transform
        self.img_transform = v2.Compose([
            v2.Resize((img_size, img_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _process_mask(self, mask):
        # Convert mask to numpy array
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        
        # Handle RGB masks
        if mask.ndim == 3:
            if "cityscapes" in self.dataset_name.lower():
                # Cityscapes: R=G=B=class_id, take first channel
                mask = mask[:, :, 0]
            elif "ADE20K" in self.dataset_name or "ade20k" in self.dataset_name.lower():
                # ADE20K: class = R + G*256 (but usually just R channel works for 150 classes)
                # Or use: class = R/10 + (G/10)*256 for some versions
                # Simplest: R channel contains class ID for most cases
                mask = mask[:, :, 0]
            else:
                # Generic: take first channel
                mask = mask[:, :, 0]
        
        mask = torch.from_numpy(mask.astype(np.int64)).long()
        
        # Dataset-specific label mapping
        if "cityscapes" in self.dataset_name.lower():
            # Cityscapes: map raw label IDs (0-33) to train IDs (0-18)
            cityscapes_map = {
                0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
                7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4,
                14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7,
                21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18,
                -1: 255, 255: 255
            }
            mask_mapped = torch.full_like(mask, 255)
            for raw_id, train_id in cityscapes_map.items():
                if raw_id >= 0:
                    mask_mapped[mask == raw_id] = train_id
            mask = mask_mapped
        
        # Clamp any remaining out-of-range values to ignore_index
        mask[mask >= self.num_classes] = self.ignore_index
        mask[mask < 0] = self.ignore_index
        
        # Resize with nearest interpolation
        mask = mask.unsqueeze(0).unsqueeze(0).float()
        mask = F.interpolate(mask, size=(self.img_size, self.img_size), mode='nearest')
        mask = mask.squeeze().long()
        
        return mask
    
    def __getitem__(self, idx):
        item = self.ds[idx]
        
        # Get image
        img = item[self.img_key]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img = img.convert("RGB")
        img = self.img_transform(img)
        
        # Get mask
        mask = item[self.mask_key]
        
        # Handle ADE20K special case (might be list of instance masks)
        if self.is_ade20k:
            if isinstance(mask, list):
                if len(mask) > 0:
                    mask = mask[0]
                else:
                    # Create empty mask if no annotations
                    mask = np.zeros((self.img_size, self.img_size), dtype=np.int64)
        
        if isinstance(mask, Image.Image):
            pass
        elif isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)
        elif isinstance(mask, dict):
            # Some datasets have mask as dict with 'bytes' key
            if 'bytes' in mask:
                import io
                mask = Image.open(io.BytesIO(mask['bytes']))
        
        mask = self._process_mask(mask)
        
        return img, mask
    
    def __len__(self):
        return len(self.ds)


# =============================================================================
# LOES Functions Adapted for Segmentation
# =============================================================================

def compute_isotropy_seg(X, eps=1e-6):
    # X: [N_samples, N_patches, D] or already flattened [N*P, D]
    if X.dim() == 3:
        X_flat = X.reshape(-1, X.shape[-1])  # [N*P, D]
    else:
        X_flat = X
    
    Xc = X_flat - X_flat.mean(0, keepdim=True)
    n_samples, n_features = Xc.shape
    
    # Use SVD for numerical stability with high-dim features
    if n_features > 512:
        try:
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


def closed_form_ridge_seg(X, Y, num_classes, reg=1e-3):
    # X: [N_samples, N_patches, D] -> flatten to [N*P, D]
    # Y: [N_samples, N_patches] -> flatten to [N*P], then one-hot to [N*P, C]
    
    if X.dim() == 3:
        X_flat = X.reshape(-1, X.shape[-1])  # [N*P, D]
    else:
        X_flat = X
    
    if Y.dim() == 2:
        Y_flat = Y.reshape(-1)  # [N*P]
    else:
        Y_flat = Y
    
    # One-hot encode
    Y_onehot = F.one_hot(Y_flat.clamp(0, num_classes-1), num_classes).float()  # [N*P, C]
    
    # Ridge regression
    Xc = X_flat - X_flat.mean(0, keepdim=True)
    Yc = Y_onehot - Y_onehot.mean(0, keepdim=True)
    
    W = torch.linalg.solve(
        Xc.t() @ Xc + reg * torch.eye(X_flat.shape[1], device=X_flat.device), 
        Xc.t() @ Yc
    )
    b = (Y_onehot.mean(0, keepdim=True) - X_flat.mean(0, keepdim=True) @ W).squeeze(0)
    
    return W, b


def mask_to_patch_labels(mask, num_patches_per_side):
    # mask: [H, W] with class indices
    # Returns: [num_patches] with majority class per patch
    
    H, W = mask.shape
    patch_h = H // num_patches_per_side
    patch_w = W // num_patches_per_side
    
    patch_labels = []
    for i in range(num_patches_per_side):
        for j in range(num_patches_per_side):
            patch = mask[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
            
            # Majority voting
            patch_flat = patch.flatten()
            if len(patch_flat) > 0:
                values, counts = torch.unique(patch_flat, return_counts=True)
                majority_class = values[counts.argmax()]
            else:
                majority_class = torch.tensor(0)
            
            patch_labels.append(majority_class)
    
    return torch.stack(patch_labels)  # [num_patches]


def collect_calibration_embeddings_seg(net, dataset, n_cal, batch_size, device="cuda"):
    net.eval()
    idx = random.sample(range(len(dataset)), min(n_cal, len(dataset)))
    loader = DataLoader(Subset(dataset, idx), batch_size, shuffle=False, num_workers=4)
    
    embeddings = []
    patch_labels_list = []
    
    # Detect num_patches from first batch
    num_patches_per_side = None
    
    with torch.no_grad():
        for imgs, masks in tqdm.tqdm(loader, desc="Collecting calibration embeddings"):
            imgs = imgs.to(device)
            
            # Get layer features: list of [B, num_patches, D]
            layer_feats = net(imgs, return_layers=True)
            
            # Detect patch grid size
            if num_patches_per_side is None:
                num_patches = layer_feats[0].shape[1]
                num_patches_per_side = int(num_patches ** 0.5)
                print(f"  Detected {num_patches} patches ({num_patches_per_side}x{num_patches_per_side})")
            
            # Convert masks to patch labels
            for mask in masks:
                pl = mask_to_patch_labels(mask, num_patches_per_side)
                patch_labels_list.append(pl)
            
            # Accumulate embeddings
            if not embeddings:
                embeddings = [[] for _ in layer_feats]
            for i, feat in enumerate(layer_feats):
                embeddings[i].append(feat.cpu())
            
            if len(patch_labels_list) >= n_cal:
                break
    
    # Concatenate
    embeddings = [torch.cat(e, dim=0)[:n_cal] for e in embeddings]  # List of [n_cal, num_patches, D]
    patch_labels = torch.stack(patch_labels_list)[:n_cal]  # [n_cal, num_patches]
    
    return embeddings, patch_labels, num_patches_per_side


def loes_select_layers_seg(embeddings, patch_labels, num_classes, K, reg=1e-3, alpha=1.0, gamma=0.5):
    # embeddings: list of [N, P, D] tensors
    # patch_labels: [N, P] tensor
    # No triangle loss for segmentation
    
    # Flatten patch_labels for one-hot encoding
    Y_flat = patch_labels.reshape(-1)  # [N*P]
    Y_onehot = F.one_hot(Y_flat.clamp(0, num_classes-1), num_classes).float()
    
    # Phase 1: Find best single layer
    best = (float("inf"), -1, None)
    for i, X in enumerate(embeddings):
        X_flat = X.reshape(-1, X.shape[-1])  # [N*P, D]
        W, b = closed_form_ridge_seg(X, patch_labels, num_classes, reg)
        loss = ((X_flat @ W + b - Y_onehot)**2).mean().item()
        iso = compute_isotropy_seg(X)
        score = loss + alpha * (1 - iso)
        if score < best[0]:
            best = (score, i, (W, b))
    
    selected = [best[1]]
    X_S = embeddings[best[1]].reshape(-1, embeddings[best[1]].shape[-1]).clone()
    y_hat = X_S @ best[2][0] + best[2][1]
    residual = Y_onehot - y_hat
    
    # Phase 2: Greedily add layers
    while len(selected) < K:
        best = (float("inf"), None, None)
        for i, X in enumerate(embeddings):
            if i in selected:
                continue
            
            X_flat = X.reshape(-1, X.shape[-1])
            Xc = X_flat - X_flat.mean(0, keepdim=True)
            XS_c = X_S - X_S.mean(0, keepdim=True)
            
            # Orthogonalize
            B_orth = torch.linalg.solve(
                XS_c.t() @ XS_c + 1e-6 * torch.eye(XS_c.shape[1], device=X_flat.device),
                XS_c.t() @ Xc
            )
            X_tilde = Xc - XS_c @ B_orth + X_flat.mean(0, keepdim=True)
            
            # Fit to residual
            W_res = torch.linalg.solve(
                X_tilde.t() @ X_tilde + reg * torch.eye(X_flat.shape[1], device=X_flat.device),
                X_tilde.t() @ residual
            )
            b_res = (residual.mean(0, keepdim=True) - X_tilde.mean(0, keepdim=True) @ W_res).squeeze(0)
            
            res_loss = ((X_tilde @ W_res + b_res - residual)**2).mean().item()
            iso = compute_isotropy_seg(X)
            
            # Redundancy
            red = max([
                (torch.norm(X_flat.t() @ embeddings[j].reshape(-1, embeddings[j].shape[-1])) / 
                 (torch.norm(X_flat) * torch.norm(embeddings[j].reshape(-1, embeddings[j].shape[-1])) + 1e-8)).item()
                for j in selected
            ])
            
            score = res_loss + alpha * (1 - iso) + gamma * red
            if score < best[0]:
                best = (score, i, (W_res, b_res, X_tilde))
        
        if best[1] is None:
            break
        
        idx = best[1]
        X_new = embeddings[idx].reshape(-1, embeddings[idx].shape[-1])
        
        # Update predictions
        W_f, b_f = closed_form_ridge_seg(embeddings[idx], patch_labels, num_classes, reg)
        y_hat = y_hat + X_new @ W_f + b_f
        residual = Y_onehot - y_hat
        
        X_S = torch.cat([X_S, X_new], dim=1)
        selected.append(idx)
    
    return selected


def compute_all_layer_scores_seg(embeddings, patch_labels, num_classes, reg=1e-3, alpha=1.0):
    # Compute LOES score for each layer independently
    Y_flat = patch_labels.reshape(-1)
    Y_onehot = F.one_hot(Y_flat.clamp(0, num_classes-1), num_classes).float()
    
    layer_scores = []
    for i, X in enumerate(embeddings):
        X_flat = X.reshape(-1, X.shape[-1])
        W, b = closed_form_ridge_seg(X, patch_labels, num_classes, reg)
        loss = ((X_flat @ W + b - Y_onehot)**2).mean().item()
        iso = compute_isotropy_seg(X)
        score = loss + alpha * (1 - iso)
        
        layer_scores.append({
            'layer_idx': i,
            'loes_score': score,
            'classification_loss': loss,
            'isotropy': iso,
            'embed_dim': X.shape[-1]
        })
    
    return layer_scores


# =============================================================================
# Segmentation Decoder
# =============================================================================

class SegmentationDecoder(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x, output_size, num_patches_per_side):
        # x: [B, num_patches, D]
        B, N, D = x.shape
        
        # Per-patch classification
        x = self.head(x)  # [B, num_patches, num_classes]
        
        # Reshape to spatial grid
        x = x.permute(0, 2, 1)  # [B, num_classes, num_patches]
        x = x.reshape(B, -1, num_patches_per_side, num_patches_per_side)  # [B, C, H_p, W_p]
        
        # Upsample to output size
        x = F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)
        
        return x  # [B, num_classes, H, W]


class LearnableWeighting(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n))
    
    def forward(self, feats_list):
        # feats_list: list of [B, P, D] tensors
        w = F.softmax(self.w, dim=0)
        out = sum(feats_list[i] * w[i] for i in range(len(feats_list)))
        return out


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_miou(preds, masks, num_classes, ignore_index=255):
    # preds: [B, H, W] predicted class indices
    # masks: [B, H, W] ground truth
    
    ious = []
    for c in range(num_classes):
        pred_c = (preds == c)
        mask_c = (masks == c)
        valid = (masks != ignore_index)
        
        intersection = ((pred_c & mask_c) & valid).sum().item()
        union = (((pred_c | mask_c) & valid).sum().item())
        
        if union > 0:
            ious.append(intersection / union)
    
    return sum(ious) / len(ious) if ious else 0.0


# =============================================================================
# Training and Evaluation
# =============================================================================

def run_eval_seg(loader, net, adapters, aggregator, decoder, topk, fusion, num_classes, 
                 num_patches_per_side, ignore_index, device):
    net.eval()
    if decoder is not None:
        decoder.eval()
    
    all_preds = []
    all_masks = []
    
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            # Get features
            layer_feats = net(imgs, return_layers=True)
            sel_feats = [layer_feats[i] for i in topk]
            
            if adapters:
                sel_feats = [adapters[i](f) for i, f in enumerate(sel_feats)]
            
            if aggregator:
                fused = aggregator(sel_feats)
            elif fusion == "concat":
                fused = torch.cat(sel_feats, dim=-1)
            elif fusion == "mean":
                fused = torch.stack(sel_feats, dim=0).mean(0)
            
            # Decode
            output_size = (masks.shape[1], masks.shape[2])
            logits = decoder(fused, output_size, num_patches_per_side)
            preds = logits.argmax(dim=1)
            
            all_preds.append(preds.cpu())
            all_masks.append(masks.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    miou = compute_miou(all_preds, all_masks, num_classes, ignore_index)
    return miou


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
    plt.title(f'{model_tag} - Layer Quality on {dataset_name} (Segmentation)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved graph to {output_path}")


# =============================================================================
# Phase Functions
# =============================================================================

def run_phase1_scoring_seg(cfg, device):
    output_dir = Path(cfg.multi_model_analysis.output_dir)
    scores_dir = output_dir / cfg.multi_model_analysis.per_layer_scores_dir
    graphs_dir = output_dir / cfg.multi_model_analysis.per_layer_graphs_dir
    
    dataset_name = cfg.dataset.name
    dataset_safe = dataset_name.split('/')[-1]
    train_ds = SegmentationDataset(dataset_name, cfg.dataset.train_split, cfg.model.img_size, cfg.dataset.num_classes)
    
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
        
        net = HFViTEncoderSeg(model_name, cfg.model.img_size).to(device)
        for p in net.parameters():
            p.requires_grad = False
        
        # Collect embeddings
        embeddings, patch_labels, num_patches_per_side = collect_calibration_embeddings_seg(
            net, train_ds, cfg.calibration.n_cal, cfg.calibration.cal_bs, device
        )
        
        # Compute scores
        layer_scores = compute_all_layer_scores_seg(embeddings, patch_labels, train_ds.num_classes)
        
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
            'embed_dim': net.embed_dim,
            'num_patches_per_side': num_patches_per_side
        }
        
        print(f"  Top-3 layers: {top_3_indices}")
        print(f"  Avg LOES score: {avg_loes:.4f}")
        
        del net, embeddings
        torch.cuda.empty_cache()
    
    # Print rankings
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
    
    rankings_csv = output_dir / cfg.multi_model_analysis.model_rankings_csv
    df_rankings = pd.DataFrame(ranking_data)
    if rankings_csv.exists():
        df_rankings.to_csv(rankings_csv, mode='a', header=False, index=False)
    else:
        df_rankings.to_csv(rankings_csv, index=False)
    
    return all_model_results


def run_phase2_training_seg(cfg, device, model_results):
    output_dir = Path(cfg.multi_model_analysis.output_dir)
    accuracies_csv = output_dir / cfg.multi_model_analysis.model_accuracies_csv
    
    dataset_name = cfg.dataset.name
    dataset_safe = dataset_name.split('/')[-1]
    
    train_ds = SegmentationDataset(dataset_name, cfg.dataset.train_split, cfg.model.img_size, cfg.dataset.num_classes)
    train_loader = DataLoader(train_ds, cfg.training.bs, shuffle=True, drop_last=True, num_workers=4)
    
    val_ds = SegmentationDataset(dataset_name, cfg.dataset.val_split, cfg.model.img_size, cfg.dataset.num_classes)
    val_loader = DataLoader(val_ds, cfg.training.test_bs, num_workers=4)
    
    print("\n" + "="*80)
    print(f"PHASE 2: TRAINING WITH GREEDY LOES TOP-3 - Dataset: {dataset_safe}")
    print("="*80)
    
    results_data = []
    
    for model_cfg in cfg.multi_model_analysis.models:
        model_name = model_cfg['name']
        model_tag = model_cfg['tag']
        
        print(f"\nTraining {model_tag}")
        
        net = HFViTEncoderSeg(model_name, cfg.model.img_size).to(device)
        for p in net.parameters():
            p.requires_grad = False
        
        # Get LOES layers
        print(f"  Running greedy LOES selection...")
        embeddings, patch_labels, num_patches_per_side = collect_calibration_embeddings_seg(
            net, train_ds, cfg.calibration.n_cal, cfg.calibration.cal_bs, device
        )
        
        topk = loes_select_layers_seg(embeddings, patch_labels, train_ds.num_classes, K=3)
        print(f"  Selected layers (greedy): {topk}")
        
        del embeddings, patch_labels
        torch.cuda.empty_cache()
        
        # WandB
        run_name = f"{model_tag}_loes_top3_{dataset_safe}"
        wandb.init(
            project=cfg.wandb.project,
            name=run_name,
            config={'model': model_tag, 'dataset': dataset_safe, 'selected_layers': topk, 'phase': 'loes_top3'},
            reinit=True
        )
        
        # Create adapters and decoder
        proj_dim = cfg.model.proj_dim
        adapters = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(net.embed_dim),
                nn.Linear(net.embed_dim, proj_dim),
                nn.GELU()
            ) for _ in topk
        ]).to(device)
        
        fused_dim = proj_dim * len(topk)
        decoder = SegmentationDecoder(fused_dim, train_ds.num_classes).to(device)
        
        # Optimizer
        params = [
            {"params": adapters.parameters(), "lr": cfg.optim.lr_probe},
            {"params": decoder.parameters(), "lr": cfg.optim.lr_probe}
        ]
        opt = torch.optim.AdamW(params, weight_decay=1e-4)
        sched = CosineAnnealingLR(opt, len(train_loader) * cfg.training.epochs, eta_min=1e-6)
        
        # Training
        best_miou = 0.0
        for ep in range(cfg.training.epochs):
            net.eval()
            decoder.train()
            
            for imgs, masks in tqdm.tqdm(train_loader, desc=f"Ep {ep}"):
                imgs, masks = imgs.to(device), masks.to(device)
                
                layer_feats = net(imgs, return_layers=True)
                sel_feats = [layer_feats[i] for i in topk]
                sel_feats = [adapters[i](f) for i, f in enumerate(sel_feats)]
                fused = torch.cat(sel_feats, dim=-1)
                
                output_size = (masks.shape[1], masks.shape[2])
                logits = decoder(fused, output_size, num_patches_per_side)
                
                loss = F.cross_entropy(logits, masks, ignore_index=train_ds.ignore_index)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                
                wandb.log({"train_loss": loss.item()})
            
            # Evaluate
            miou = run_eval_seg(val_loader, net, adapters, None, decoder, topk, "concat",
                               train_ds.num_classes, num_patches_per_side, train_ds.ignore_index, device)
            if miou > best_miou:
                best_miou = miou
            
            wandb.log({"val_miou": miou, "epoch": ep})
            print(f"  Ep {ep} mIoU: {miou:.4f}")
        
        wandb.finish()
        
        results_data.append({
            'model': model_tag,
            'dataset': dataset_safe,
            'phase': 'loes_top3',
            'selected_layers': str(topk),
            'val_miou': best_miou
        })
        
        print(f"  Best mIoU: {best_miou:.4f}")
        
        del net, decoder, adapters
        torch.cuda.empty_cache()
    
    # Save results
    df_results = pd.DataFrame(results_data)
    if accuracies_csv.exists():
        df_results.to_csv(accuracies_csv, mode='a', header=False, index=False)
    else:
        df_results.to_csv(accuracies_csv, index=False)


def run_phase3_lastlayer_seg(cfg, device):
    output_dir = Path(cfg.multi_model_analysis.output_dir)
    baseline_csv = output_dir / cfg.multi_model_analysis.last_layer_baseline_csv
    
    dataset_name = cfg.dataset.name
    dataset_safe = dataset_name.split('/')[-1]
    
    train_ds = SegmentationDataset(dataset_name, cfg.dataset.train_split, cfg.model.img_size, cfg.dataset.num_classes)
    train_loader = DataLoader(train_ds, cfg.training.bs, shuffle=True, drop_last=True, num_workers=4)
    
    val_ds = SegmentationDataset(dataset_name, cfg.dataset.val_split, cfg.model.img_size, cfg.dataset.num_classes)
    val_loader = DataLoader(val_ds, cfg.training.test_bs, num_workers=4)
    
    print("\n" + "="*80)
    print(f"PHASE 3: LAST LAYER BASELINE - Dataset: {dataset_safe}")
    print("="*80)
    
    baseline_data = []
    
    for model_cfg in cfg.multi_model_analysis.models:
        model_name = model_cfg['name']
        model_tag = model_cfg['tag']
        
        print(f"\nTraining {model_tag} (last layer only)")
        
        net = HFViTEncoderSeg(model_name, cfg.model.img_size).to(device)
        for p in net.parameters():
            p.requires_grad = False
        
        # Detect num_patches
        with torch.no_grad():
            dummy = torch.randn(1, 3, cfg.model.img_size, cfg.model.img_size).to(device)
            layer_feats = net(dummy, return_layers=True)
            num_patches_per_side = int(layer_feats[0].shape[1] ** 0.5)
        
        topk = [net.n_layers - 1]
        print(f"  Using layer: {topk[0]}")
        
        # WandB
        run_name = f"{model_tag}_lastlayer_{dataset_safe}"
        wandb.init(
            project=cfg.wandb.project,
            name=run_name,
            config={'model': model_tag, 'dataset': dataset_safe, 'layer': topk[0], 'phase': 'last_layer'},
            reinit=True
        )
        
        # Create adapter and decoder
        proj_dim = cfg.model.proj_dim
        adapters = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(net.embed_dim),
                nn.Linear(net.embed_dim, proj_dim),
                nn.GELU()
            )
        ]).to(device)
        
        decoder = SegmentationDecoder(proj_dim, train_ds.num_classes).to(device)
        
        # Optimizer
        params = [
            {"params": adapters.parameters(), "lr": cfg.optim.lr_probe},
            {"params": decoder.parameters(), "lr": cfg.optim.lr_probe}
        ]
        opt = torch.optim.AdamW(params, weight_decay=1e-4)
        sched = CosineAnnealingLR(opt, len(train_loader) * cfg.training.epochs, eta_min=1e-6)
        
        # Training
        best_miou = 0.0
        for ep in range(cfg.training.epochs):
            net.eval()
            decoder.train()
            
            for imgs, masks in tqdm.tqdm(train_loader, desc=f"Ep {ep}"):
                imgs, masks = imgs.to(device), masks.to(device)
                
                layer_feats = net(imgs, return_layers=True)
                sel_feats = [layer_feats[topk[0]]]
                sel_feats = [adapters[0](sel_feats[0])]
                fused = sel_feats[0]
                
                output_size = (masks.shape[1], masks.shape[2])
                logits = decoder(fused, output_size, num_patches_per_side)
                
                loss = F.cross_entropy(logits, masks, ignore_index=train_ds.ignore_index)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                
                wandb.log({"train_loss": loss.item()})
            
            # Evaluate
            miou = run_eval_seg(val_loader, net, adapters, None, decoder, topk, "concat",
                               train_ds.num_classes, num_patches_per_side, train_ds.ignore_index, device)
            if miou > best_miou:
                best_miou = miou
            
            wandb.log({"val_miou": miou, "epoch": ep})
            print(f"  Ep {ep} mIoU: {miou:.4f}")
        
        wandb.finish()
        
        baseline_data.append({
            'model': model_tag,
            'dataset': dataset_safe,
            'layer_idx': topk[0],
            'val_miou': best_miou
        })
        
        print(f"  Best mIoU: {best_miou:.4f}")
        
        del net, decoder, adapters
        torch.cuda.empty_cache()
    
    # Save results
    df_baseline = pd.DataFrame(baseline_data)
    if baseline_csv.exists():
        df_baseline.to_csv(baseline_csv, mode='a', header=False, index=False)
    else:
        df_baseline.to_csv(baseline_csv, index=False)


def run_phase4_learnable_weight_seg(cfg, device):
    output_dir = Path(cfg.multi_model_analysis.output_dir)
    learnable_csv = output_dir / cfg.multi_model_analysis.learnable_weight_csv
    
    dataset_name = cfg.dataset.name
    dataset_safe = dataset_name.split('/')[-1]
    
    train_ds = SegmentationDataset(dataset_name, cfg.dataset.train_split, cfg.model.img_size, cfg.dataset.num_classes)
    train_loader = DataLoader(train_ds, cfg.training.bs, shuffle=True, drop_last=True, num_workers=4)
    
    val_ds = SegmentationDataset(dataset_name, cfg.dataset.val_split, cfg.model.img_size, cfg.dataset.num_classes)
    val_loader = DataLoader(val_ds, cfg.training.test_bs, num_workers=4)
    
    print("\n" + "="*80)
    print(f"PHASE 4: LEARNABLE WEIGHTING (ALL LAYERS) - Dataset: {dataset_safe}")
    print("="*80)
    
    learnable_data = []
    
    for model_cfg in cfg.multi_model_analysis.models:
        model_name = model_cfg['name']
        model_tag = model_cfg['tag']
        
        print(f"\nTraining {model_tag} (learnable weighting)")
        
        net = HFViTEncoderSeg(model_name, cfg.model.img_size).to(device)
        for p in net.parameters():
            p.requires_grad = False
        
        # Detect num_patches and layers
        with torch.no_grad():
            dummy = torch.randn(1, 3, cfg.model.img_size, cfg.model.img_size).to(device)
            layer_feats = net(dummy, return_layers=True)
            num_patches_per_side = int(layer_feats[0].shape[1] ** 0.5)
            n_layers = len(layer_feats)
        
        topk = list(range(n_layers))
        print(f"  Using all {n_layers} layers with learnable weights")
        
        # WandB
        run_name = f"{model_tag}_learnable_{dataset_safe}"
        wandb.init(
            project=cfg.wandb.project,
            name=run_name,
            config={'model': model_tag, 'dataset': dataset_safe, 'n_layers': n_layers, 'phase': 'learnable_weight'},
            reinit=True
        )
        
        # Create adapters, aggregator, and decoder
        proj_dim = cfg.model.proj_dim
        adapters = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(net.embed_dim),
                nn.Linear(net.embed_dim, proj_dim),
                nn.GELU()
            ) for _ in topk
        ]).to(device)
        
        aggregator = LearnableWeighting(n_layers).to(device)
        decoder = SegmentationDecoder(proj_dim, train_ds.num_classes).to(device)
        
        # Optimizer
        params = [
            {"params": adapters.parameters(), "lr": cfg.optim.lr_probe},
            {"params": aggregator.parameters(), "lr": 1e-3},
            {"params": decoder.parameters(), "lr": cfg.optim.lr_probe}
        ]
        opt = torch.optim.AdamW(params, weight_decay=1e-4)
        sched = CosineAnnealingLR(opt, len(train_loader) * cfg.training.epochs, eta_min=1e-6)
        
        # Training
        best_miou = 0.0
        for ep in range(cfg.training.epochs):
            net.eval()
            decoder.train()
            aggregator.train()
            
            for imgs, masks in tqdm.tqdm(train_loader, desc=f"Ep {ep}"):
                imgs, masks = imgs.to(device), masks.to(device)
                
                layer_feats = net(imgs, return_layers=True)
                sel_feats = [adapters[i](layer_feats[i]) for i in topk]
                fused = aggregator(sel_feats)
                
                output_size = (masks.shape[1], masks.shape[2])
                logits = decoder(fused, output_size, num_patches_per_side)
                
                loss = F.cross_entropy(logits, masks, ignore_index=train_ds.ignore_index)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                
                wandb.log({"train_loss": loss.item()})
            
            # Evaluate
            miou = run_eval_seg(val_loader, net, adapters, aggregator, decoder, topk, "learnable",
                               train_ds.num_classes, num_patches_per_side, train_ds.ignore_index, device)
            if miou > best_miou:
                best_miou = miou
            
            # Log learned weights
            weights = F.softmax(aggregator.w, dim=0).detach().cpu().numpy()
            for i, w in enumerate(weights):
                wandb.log({f"layer_{i}_weight": w})
            
            wandb.log({"val_miou": miou, "epoch": ep})
            print(f"  Ep {ep} mIoU: {miou:.4f}")
        
        wandb.finish()
        
        # Get final weights
        final_weights = F.softmax(aggregator.w, dim=0).detach().cpu().numpy()
        top_weighted_layers = np.argsort(final_weights)[-3:][::-1].tolist()
        
        learnable_data.append({
            'model': model_tag,
            'dataset': dataset_safe,
            'n_layers': n_layers,
            'top_weighted_layers': str(top_weighted_layers),
            'val_miou': best_miou
        })
        
        print(f"  Best mIoU: {best_miou:.4f}")
        print(f"  Top weighted layers: {top_weighted_layers}")
        
        del net, decoder, adapters, aggregator
        torch.cuda.empty_cache()
    
    # Save results
    df_learnable = pd.DataFrame(learnable_data)
    if learnable_csv.exists():
        df_learnable.to_csv(learnable_csv, mode='a', header=False, index=False)
    else:
        df_learnable.to_csv(learnable_csv, index=False)


def run_multi_model_analysis_seg(cfg, device):
    print("\n" + "="*80)
    print("STARTING MULTI-MODEL SEGMENTATION ANALYSIS")
    print("="*80)
    print(f"Dataset: {cfg.dataset.name}")
    print(f"Models: {len(cfg.multi_model_analysis.models)}")
    print(f"Output: {cfg.multi_model_analysis.output_dir}")
    print("="*80)
    
    model_results = None
    # Phase 1: Score all layers
    if cfg.multi_model_analysis.run_phase1:
     model_results = run_phase1_scoring_seg(cfg, device)
    
    # Phase 2: Train LOES top-3
    run_phase2_training_seg(cfg, device, model_results)
    
    # Phase 3: Last layer baseline
    run_phase3_lastlayer_seg(cfg, device)
    
    # Phase 4: Learnable weighting
    run_phase4_learnable_weight_seg(cfg, device)
    
    print("\n" + "="*80)
    print("MULTI-MODEL SEGMENTATION ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Results saved to: {cfg.multi_model_analysis.output_dir}")


@hydra.main(config_path="conf", config_name="config_seg", version_base=None)
def main(cfg: DictConfig):
    device = cfg.device
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    
    if cfg.multi_model_analysis.enabled:
        print("\n" + "="*80)
        print("MULTI-MODEL SEGMENTATION ANALYSIS")
        print("="*80)
        run_multi_model_analysis_seg(cfg, device)
        return
    
    print("Single model training not implemented for segmentation yet.")
    print("Use multi_model_analysis.enabled=true")


if __name__ == "__main__":
    main()