#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="0"

# ------------------
# DATASETS FOR SEGMENTATION
# ------------------
# Format: "dataset_name|train_split|val_split|num_classes"
DATASETS=(
  "Chris1/cityscapes_segmentation|train|validation|19"
  "EduardoPacheco/FoodSeg103|train|validation|104"
  "GATE-engine/COCOStuff164K|train|val|171"
)

# Note: ADE20K commented out until we verify mask format
# "1aurent/ADE20K|train|validation|150"

EPOCHS=20
CMD="python main_image_seg.py"

echo "========================================================"
echo "MULTI-MODEL SEGMENTATION ANALYSIS"
echo "========================================================"
echo "Models: DINOv2, DINOv3, CLIPSeg, BEiT"
echo "Datasets: ${#DATASETS[@]}"
echo "Phases: 1) Scoring  2) LOES Top-3  3) Last Layer  4) Learnable Weights"
echo "========================================================"

for ds_entry in "${DATASETS[@]}"; do
    IFS='|' read -r ds_name ds_train ds_val num_classes <<< "${ds_entry}"
    ds_safe=$(echo "$ds_name" | cut -d'/' -f2)
    
    echo ""
    echo "========================================================"
    echo "DATASET: $ds_safe (${num_classes} classes)"
    echo "========================================================"
    
    $CMD \
        multi_model_analysis.enabled=true \
        dataset.name="$ds_name" \
        dataset.train_split="$ds_train" \
        dataset.val_split="$ds_val" \
        dataset.num_classes=$num_classes \
        training.epochs=$EPOCHS \
        wandb.project="ICML_LOES_Segmentation_11pm"
    
    echo ""
    echo "Completed $ds_safe"
    echo "========================================================"
done

echo ""
echo "========================================================"
echo "ALL SEGMENTATION EXPERIMENTS COMPLETED!"
echo "========================================================"
echo "Results in: seg_multi_model_analysis/"
echo "  - per_layer_scores/     (CSV files)"
echo "  - per_layer_graphs/     (PNG graphs)"
echo "  - model_rankings.csv    (Top-3 scores)"
echo "  - model_accuracies.csv  (LOES Top-3 mIoU)"
echo "  - last_layer_baseline.csv (Last layer mIoU)"
echo "  - learnable_weight.csv  (Learnable weighting mIoU)"
echo "========================================================"