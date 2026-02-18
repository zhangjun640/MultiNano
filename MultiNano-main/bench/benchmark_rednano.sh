#!/bin/bash

# ==============================================================================
# RedNano m6A Detection Workflow Demo
# ==============================================================================
# Description: 
#   This script demonstrates the full workflow of RedNano for m6A detection.
#   It processes data step-by-step for clear visualization on GitHub.
# ==============================================================================

# --- Configuration (Environment & Paths) ---

# 1. Base Setup 
BASE_DIR=$(pwd)
OUTPUT_DIR="${BASE_DIR}/rednano_demo_output"
mkdir -p "$OUTPUT_DIR"


SCRIPT_EXTRACT="${BASE_DIR}/RedNano/scripts/extract_features.py"
SCRIPT_PREDICT="${BASE_DIR}/RedNano/scripts/predict.py"
SCRIPT_AGGREGATE="${BASE_DIR}/scripts/aggre_features_to_site_level.py"

# 3. Model Paths 
MODEL_FILE="${BASE_DIR}/RedNano-nf/RedNano-main/models/mil_allspecies_model_states.pt"


INPUT_TOMBO_DIR="${BASE_DIR}/demo/tombo_results" 
REFERENCE_FASTA="${BASE_DIR}/demo/cc.fasta"

# 5. Runtime Parameters
THREADS=16         # CPU threads 
GPU_DEVICE=0       # Target GPU ID 

# Log file 
LOG_FILE="${OUTPUT_DIR}/rednano_execution.log"

# Clear previous logs
> "$LOG_FILE"


# ==============================================================================
# Step 1: Feature Extraction
# ==============================================================================
print_header "Step 1: Feature Extraction "


FEATURES_H5="${OUTPUT_DIR}/extracted_features.h5"

# Execution 
python3 "$SCRIPT_EXTRACT" \
    --input "$INPUT_TOMBO_DIR" \
    --reference "$REFERENCE_FASTA" \
    --output "$FEATURES_H5" \
    --cpu "$THREADS" \
    2>&1 | tee -a "$LOG_FILE"

log_msg ">>> Step 1 Completed. "


# ==============================================================================
# Step 2: Model Prediction (Read-Level)
# ==============================================================================
print_header "Step 2: Model Prediction "

PRED_RESULTS="${OUTPUT_DIR}/read_level_predictions.txt"
export CUDA_VISIBLE_DEVICES=$GPU_DEVICE

# Execution 
python3 "$SCRIPT_PREDICT" \
    --model_path "$MODEL_FILE" \
    --input_features "$FEATURES_H5" \
    --output "$PRED_RESULTS" \
    --batch_size 1024 \
    2>&1 | tee -a "$LOG_FILE"

log_msg ">>> Step 2 Completed. "


# ==============================================================================
# Step 3: Site-Level Aggregation
# ==============================================================================
print_header "Step 3: Site-Level Aggregation"

SITE_LEVEL_OUT="${OUTPUT_DIR}/site_level_m6A_results.txt"

# Execution 
python3 "$SCRIPT_AGGREGATE" \
    --input "$PRED_RESULTS" \
    --output "$SITE_LEVEL_OUT" \
    2>&1 | tee -a "$LOG_FILE"

log_msg ">>> Step 3 Completed. "


BASE_DIR=$(pwd)
TRAIN_SCRIPT="${BASE_DIR}/RedNano/scripts/train.py"
TRAIN_DATA="${BASE_DIR}/data/train_features.h5"
VALID_DATA="${BASE_DIR}/data/valid_features.h5"
SAVE_DIR="${BASE_DIR}/retrained_rednano"
mkdir -p "$SAVE_DIR"

# --- Execution / 执行 ---
echo "Starting RedNano Training... / 开始 RedNano 训练..."
python3 "$TRAIN_SCRIPT" \
    --train_path "$TRAIN_DATA" \
    --valid_path "$VALID_DATA" \
    --save_path "$SAVE_DIR" \
    --model_type "mil" \
    --batch_size 64 \
    --epochs 20