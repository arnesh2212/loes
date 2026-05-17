
#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="3"

# ------------------
# 1. GRID DEFINITIONS
# ------------------

KS=(2 3 4)

MODELS=(
  "vit_small_patch14_dinov2.lvd142m"
  # "vit_base_patch16_224.mae"
)
# Datasets: hf_id|train_split|test_split
DATASETS=(
  "ILSVRC/imagenet-1k|train|validation|test"
  "saberzl/SID_Set|train|validation|"
  # "1aurent/PatchCamelyon|train|valid|test"
  # "matthieulel/galaxy10_decals|train|test|"
  # "flwrlabs/fed-isic2019|train|test|"
  # "clane9/imagenet-100|train|validation|"
  # "tanganke/stanford_cars|train|test|"
  # "timm/mini-imagenet|train|validation|"
  # "maurice-fp/stanford-dogs|train|test|"
  # "bentrevett/caltech-ucsd-birds-200-2011|train|test|"
  # "randall-lab/dtd|train|validation|"
  # "uoft-cs/cifar100|train|test|"
  # "tanganke/sun397|train|test|"


)


from datasets import load_dataset

ds = load_dataset(
    "ILSVRC/imagenet-1k",
      download_mode="reuse_cache_if_exists"
)

ds.save_to_disk("imagenet_train")


# Hyperparameters
EPOCHS=15
BS=128
CAL_SIZE=2000
CMD="python main_image.py"

echo "========================================================"
echo "Starting Full ICML Grid Search"
echo "Models: ${#MODELS[@]} | Datasets: ${#DATASETS[@]}"
echo "========================================================"

# ------------------
# 2. MAIN LOOP
# ------------------

for model in "${MODELS[@]}"; do
  for ds_entry in "${DATASETS[@]}"; do
    # 1. READ ALL 4 VARIABLES
    IFS='|' read -r ds_name ds_train ds_val ds_test <<< "${ds_entry}"
    
    ds_safe=$(echo "$ds_name" | cut -d'/' -f2)
    
    echo "--------------------------------------------------------"
    echo "Current: $model | $ds_safe"
    echo "Splits: Train=$ds_train, Val=$ds_val, Test=$ds_test"
    echo "--------------------------------------------------------"

    echo ">>> [Baseline] Last Layer on $ds_safe"
    $CMD \
        selection.mode="last" topk=1 \
        ablation.fusion="concat" ablation.use_geo_loss=false ablation.no_adapters=false \
        model.name="$model" dataset.name="$ds_name" \
        dataset.train_split="$ds_train" dataset.val_split="$ds_val" dataset.test_split="$ds_test" \
        training.epochs=$EPOCHS

    # ====================================================
    # A. MAIN PROPOSAL (LOES + GCR + Adapters)
    # ====================================================
    for k in "${KS[@]}"; do
        echo ">>> [Proposal] LOES (k=$k) on $ds_safe"
        $CMD \
            selection.mode="loes" topk=$k \
            ablation.fusion="concat" ablation.use_geo_loss=true ablation.no_adapters=false \
            model.name="$model" dataset.name="$ds_name" \
            dataset.train_split="$ds_train" dataset.val_split="$ds_val" dataset.test_split="$ds_test" \
            training.epochs=$EPOCHS training.bs=$BS calibration.n_cal=$CAL_SIZE
    done

    # ====================================================
    # B. BASELINE: RANDOM SELECTION
    # ====================================================
    for k in "${KS[@]}"; do
        echo ">>> [Baseline] Random (k=$k) on $ds_safe"
        $CMD \
            selection.mode="random" topk=$k \
            ablation.fusion="concat" ablation.use_geo_loss=true ablation.no_adapters=false \
            model.name="$model" dataset.name="$ds_name" \
            dataset.train_split="$ds_train" dataset.val_split="$ds_val" dataset.test_split="$ds_test" \
            training.epochs=$EPOCHS training.bs=$BS calibration.n_cal=$CAL_SIZE
    done

    # ====================================================
    # C. BASELINE: LEARNABLE WEIGHTING & LAST LAYER
    # ====================================================
    echo ">>> [Baseline] Learnable Weighting on $ds_safe"
    $CMD \
        selection.mode="learnable_weight" topk=0 \
        ablation.fusion="sum" ablation.use_geo_loss=true ablation.no_adapters=false \
        model.name="$model" dataset.name="$ds_name" \
        dataset.train_split="$ds_train" dataset.val_split="$ds_val" dataset.test_split="$ds_test" \
        training.epochs=$EPOCHS


    # ====================================================
    # D. ABLATIONS (Structural)
    # ====================================================
    
    echo ">>> [Ablation] No Geo Loss (k=3)"
    $CMD \
        selection.mode="loes" topk=3 \
        ablation.fusion="concat" ablation.use_geo_loss=false ablation.no_adapters=false \
        model.name="$model" dataset.name="$ds_name" \
        dataset.train_split="$ds_train" dataset.val_split="$ds_val" dataset.test_split="$ds_test" \
        training.epochs=$EPOCHS

    echo ">>> [Ablation] Mean Fusion (k=3)"
    $CMD \
        selection.mode="loes" topk=3 \
        ablation.fusion="mean" ablation.use_geo_loss=true ablation.no_adapters=false \
        model.name="$model" dataset.name="$ds_name" \
        dataset.train_split="$ds_train" dataset.val_split="$ds_val" dataset.test_split="$ds_test" \
        training.epochs=$EPOCHS


  done
done