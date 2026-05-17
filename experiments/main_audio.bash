#!/usr/bin/env bash
set -euo pipefail

# Set the GPU to use
export CUDA_VISIBLE_DEVICES="2"

# ------------------
# 1. GRID DEFINITIONS
# ------------------

# K values for LOES and Random Selection (Same logic as image)#!/usr/bin/env bash
set -euo pipefail

# Set the GPU to use
export CUDA_VISIBLE_DEVICES="1"

# ------------------
# 1. GRID DEFINITIONS
# ------------------

# K values for LOES and Random Selection (Same logic as image)
KS=(2 3 4)

# Models (Hugging Face Model IDs for Audio)
# Wav2Vec2 is the standard backbone here.
MODELS=(
  "facebook/wav2vec2-base-960h"
  # "facebook/wav2vec2-large-xlsr-53" # Uncomment for multilingual/larger tests
  # "microsoft/wavlm-base-plus"       # Alternative SOTA backbone
)

# Datasets: hf_id|train_split|test_split
# Using standard HF audio classification benchmarks
DATASETS=(
#   "alexnasa/english_accents|train|test"        # Keyword Spotting (v0.02 usually default)
  "marsyas/gtzan|train|test"        # Music Genre Classification
  # "superb|train|test"               # Valid only if main_audio.py handles specific subset config
  # "audiofolder|train|test"          # If you have local custom data
)

# Hyperparameters
# Audio models are often heavier on VRAM than ViT-Small. 
# Reduced BS from 128 -> 32/64 to prevent OOM.
EPOCHS=15
BS=32 
CAL_SIZE=1000  # Slightly lower calibration size usually suffices for audio
CMD="python main.py" 

echo "========================================================"
echo "Starting Full ICML Grid Search (AUDIO)"
echo "Models: ${#MODELS[@]} | Datasets: ${#DATASETS[@]}"
echo "========================================================"

# ------------------
# 2. MAIN LOOP
# ------------------

for model in "${MODELS[@]}"; do
  for ds_entry in "${DATASETS[@]}"; do
    IFS='|' read -r ds_name ds_train ds_test <<< "${ds_entry}"
    
    # Clean name for logging (e.g. facebook/wav2vec... -> wav2vec...)
    ds_safe=$(echo "$ds_name" | cut -d'/' -f2)
    
    echo "--------------------------------------------------------"
    echo "Current: $model | $ds_safe"
    echo "--------------------------------------------------------"

    # ====================================================
    # A. MAIN PROPOSAL (LOES + GCR + Adapters)
    #    Run for all k=[2, 3, 4]
    # ====================================================
    for k in "${KS[@]}"; do
        echo ">>> [Proposal] LOES (k=$k) on $ds_safe"
        $CMD \
            selection.mode="loes" topk=$k \
            ablation.fusion="concat" ablation.use_geo_loss=true ablation.no_adapters=false \
            model.name="$model" dataset.name="$ds_name" \
            dataset.train_split="$ds_train" dataset.test_split="$ds_test" \
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
            dataset.train_split="$ds_train" dataset.test_split="$ds_test" \
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
        dataset.train_split="$ds_train" dataset.test_split="$ds_test" \
        training.epochs=$EPOCHS training.bs=$BS

    echo ">>> [Baseline] Last Layer on $ds_safe"
    $CMD \
        selection.mode="last" topk=1 \
        ablation.fusion="concat" ablation.use_geo_loss=false ablation.no_adapters=false \
        model.name="$model" dataset.name="$ds_name" \
        dataset.train_split="$ds_train" dataset.test_split="$ds_test" \
        training.epochs=$EPOCHS training.bs=$BS

    # ====================================================
    # D. ABLATIONS (Structural) - Run on k=3
    # ====================================================
    
    # 1. No Geometric Consistency Loss
    echo ">>> [Ablation] No Geo Loss (k=3)"
    $CMD \
        selection.mode="loes" topk=3 \
        ablation.fusion="concat" ablation.use_geo_loss=false ablation.no_adapters=false \
        model.name="$model" dataset.name="$ds_name" \
        dataset.train_split="$ds_train" dataset.test_split="$ds_test" \
        training.epochs=$EPOCHS training.bs=$BS

    # 2. Mean Fusion
    echo ">>> [Ablation] Mean Fusion (k=3)"
    $CMD \
        selection.mode="loes" topk=3 \
        ablation.fusion="mean" ablation.use_geo_loss=true ablation.no_adapters=false \
        model.name="$model" dataset.name="$ds_name" \
        dataset.train_split="$ds_train" dataset.test_split="$ds_test" \
        training.epochs=$EPOCHS training.bs=$BS

    # 3. No Adapters
    echo ">>> [Ablation] No Adapters (k=3)"
    $CMD \
        selection.mode="loes" topk=3 \
        ablation.fusion="concat" ablation.use_geo_loss=true ablation.no_adapters=true \
        model.name="$model" dataset.name="$ds_name" \
        dataset.train_split="$ds_train" dataset.test_split="$ds_test" \
        training.epochs=$EPOCHS training.bs=$BS

  done
done

echo "========================================================"
echo "Full Audio Grid Search Complete."
echo "========================================================"
KS=(2 3 4)

# Models (Hugging Face Model IDs for Audio)
# Wav2Vec2 is the standard backbone here.
MODELS=(
  "facebook/wav2vec2-base-960h"
  # "facebook/wav2vec2-large-xlsr-53" # Uncomment for multilingual/larger tests
  # "microsoft/wavlm-base-plus"       # Alternative SOTA backbone
)

# Datasets: hf_id|train_split|test_split
# Using standard HF audio classification benchmarks
DATASETS=(
  "speech_commands|train|test"        # Keyword Spotting (v0.02 usually default)
  # "marsyas/gtzan|train|test"        # Music Genre Classification
  # "superb|train|test"               # Valid only if main_audio.py handles specific subset config
  # "audiofolder|train|test"          # If you have local custom data
)

# Hyperparameters
# Audio models are often heavier on VRAM than ViT-Small. 
# Reduced BS from 128 -> 32/64 to prevent OOM.
EPOCHS=15
BS=32 
CAL_SIZE=1000  # Slightly lower calibration size usually suffices for audio
CMD="python main_audio.py" 

echo "========================================================"
echo "Starting Full ICML Grid Search (AUDIO)"
echo "Models: ${#MODELS[@]} | Datasets: ${#DATASETS[@]}"
echo "========================================================"

# ------------------
# 2. MAIN LOOP
# ------------------

for model in "${MODELS[@]}"; do
  for ds_entry in "${DATASETS[@]}"; do
    IFS='|' read -r ds_name ds_train ds_test <<< "${ds_entry}"
    
    # Clean name for logging (e.g. facebook/wav2vec... -> wav2vec...)
    ds_safe=$(echo "$ds_name" | cut -d'/' -f2)
    
    echo "--------------------------------------------------------"
    echo "Current: $model | $ds_safe"
    echo "--------------------------------------------------------"

    # ====================================================
    # A. MAIN PROPOSAL (LOES + GCR + Adapters)
    #    Run for all k=[2, 3, 4]
    # ====================================================
    for k in "${KS[@]}"; do
        echo ">>> [Proposal] LOES (k=$k) on $ds_safe"
        $CMD \
            selection.mode="loes" topk=$k \
            ablation.fusion="concat" ablation.use_geo_loss=true ablation.no_adapters=false \
            model.name="$model" dataset.name="$ds_name" \
            dataset.train_split="$ds_train" dataset.test_split="$ds_test" \
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
            dataset.train_split="$ds_train" dataset.test_split="$ds_test" \
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
        dataset.train_split="$ds_train" dataset.test_split="$ds_test" \
        training.epochs=$EPOCHS training.bs=$BS

    echo ">>> [Baseline] Last Layer on $ds_safe"
    $CMD \
        selection.mode="last" topk=1 \
        ablation.fusion="concat" ablation.use_geo_loss=false ablation.no_adapters=false \
        model.name="$model" dataset.name="$ds_name" \
        dataset.train_split="$ds_train" dataset.test_split="$ds_test" \
        training.epochs=$EPOCHS training.bs=$BS

    # ====================================================
    # D. ABLATIONS (Structural) - Run on k=3
    # ====================================================
    
    # 1. No Geometric Consistency Loss
    echo ">>> [Ablation] No Geo Loss (k=3)"
    $CMD \
        selection.mode="loes" topk=3 \
        ablation.fusion="concat" ablation.use_geo_loss=false ablation.no_adapters=false \
        model.name="$model" dataset.name="$ds_name" \
        dataset.train_split="$ds_train" dataset.test_split="$ds_test" \
        training.epochs=$EPOCHS training.bs=$BS

    # 2. Mean Fusion
    echo ">>> [Ablation] Mean Fusion (k=3)"
    $CMD \
        selection.mode="loes" topk=3 \
        ablation.fusion="mean" ablation.use_geo_loss=true ablation.no_adapters=false \
        model.name="$model" dataset.name="$ds_name" \
        dataset.train_split="$ds_train" dataset.test_split="$ds_test" \
        training.epochs=$EPOCHS training.bs=$BS

    # 3. No Adapters
    echo ">>> [Ablation] No Adapters (k=3)"
    $CMD \
        selection.mode="loes" topk=3 \
        ablation.fusion="concat" ablation.use_geo_loss=true ablation.no_adapters=true \
        model.name="$model" dataset.name="$ds_name" \
        dataset.train_split="$ds_train" dataset.test_split="$ds_test" \
        training.epochs=$EPOCHS training.bs=$BS

  done
done

echo "========================================================"
echo "Full Audio Grid Search Complete."
echo "========================================================"
