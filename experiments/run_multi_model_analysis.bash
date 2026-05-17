#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="2"

# ------------------
# DATASETS
# ------------------
DATASETS=(
  "tanganke/stanford_cars|train||test"
  "timm/mini-imagenet|train|validation|"
  "maurice-fp/stanford-dogs|train||test"
  "bentrevett/caltech-ucsd-birds-200-2011|train||test"
  "randall-lab/dtd|train|validation|"
  "uoft-cs/cifar100|train||test"
  "tanganke/sun397|train||test"
  "clane9/imagenet-100|train|validation|"
)


# Hyperparameters
EPOCHS=15
CMD="python main_image.py"

echo "========================================================"
echo "MULTI-MODEL ANALYSIS"
echo "========================================================"
echo "Models: CLIP, MAE, DINOv2, DINOv3, ImageNet21k"
echo "Datasets: ${#DATASETS[@]}"
echo "Phases: 1) Scoring  2) Top-3 Training  3) Last Layer Baseline"
echo "========================================================"

for ds_entry in "${DATASETS[@]}"; do
    IFS='|' read -r ds_name ds_train ds_val ds_test <<< "${ds_entry}"
    ds_safe=$(echo "$ds_name" | cut -d'/' -f2)
    
    echo ""
    echo "========================================================"
    echo "DATASET: $ds_safe"
    echo "========================================================"
    
    $CMD \
        multi_model_analysis.enabled=true \
        dataset.name="$ds_name" \
        dataset.train_split="$ds_train" \
        dataset.val_split="$ds_val" \
        dataset.test_split="$ds_test" \
        training.epochs=$EPOCHS \
        wandb.project="ICML_LOES_6_HF_Models_Final"
    
    echo ""
    echo "Completed $ds_safe"
    echo "========================================================"
done

echo ""
echo "========================================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "========================================================"
echo "Results in: multi_model_analysis/"
echo "  - per_layer_scores/     (CSV files)"
echo "  - per_layer_graphs/     (PNG graphs)"
echo "  - model_rankings.csv    (Top-3 scores)"
echo "  - model_accuracies.csv  (Training results)"
echo "  - last_layer_baseline.csv (Baseline)"
echo "========================================================"