#!/bin/bash

# ==============================================================================
# nanom6A Detection Workflow Demo
# ==============================================================================
# Description: 
#   This script demonstrates the full workflow of nanom6A for m6A detection.
#   It relies on features extracted from Nanopolish eventalign output.
# ==============================================================================

# --- Configuration (Environment & Paths) ---

# 1. Base Setup 
BASE_DIR=$(pwd)
OUTPUT_DIR="${BASE_DIR}/nanom6a_demo_output"
mkdir -p "$OUTPUT_DIR"

# 2. Script Paths 
SCRIPT_EXTRACT="${BASE_DIR}/extract_raw_and_feature_fast.py"
SCRIPT_PREDICT="${BASE_DIR}/predict_sites.py"
SCRIPT_FILTER="${BASE_DIR}/fillter_output.py"


MODEL_FILE="${BASE_DIR}/model/model.pkl"

# Input file: Output from 'nanopolish eventalign'
INPUT_EVENTALIGN="${BASE_DIR}/test_data/eventalign.txt"

# 4. Runtime Parameters
CPU_THREADS=16     # CPU threads 
PROB_THRESHOLD=0.5 # Probability threshold for filtering

# Log file
LOG_FILE="${OUTPUT_DIR}/nanom6a_execution.log"

# Clear previous logs
> "$LOG_FILE"


# ==============================================================================
# Step 1: Feature Extraction
# ==============================================================================
print_header "Step 1: Feature Extraction "

EXTRACTED_FEATURES="${OUTPUT_DIR}/extracted_features.tsv"

# Execution 
python3 "$SCRIPT_EXTRACT" \
    --input "$INPUT_EVENTALIGN" \
    --out_file "$EXTRACTED_FEATURES" \
    2>&1 | tee -a "$LOG_FILE"

log_msg ">>> Step 1 Completed. "


# ==============================================================================
# Step 2: Site-Level Prediction
# ==============================================================================
print_header "Step 2: Site-Level Prediction "

PRED_RESULTS="${OUTPUT_DIR}/raw_predictions.txt"

# Execution 
python3 "$SCRIPT_PREDICT" \
    --input "$EXTRACTED_FEATURES" \
    --model "$MODEL_FILE" \
    --out_file "$PRED_RESULTS" \
    2>&1 | tee -a "$LOG_FILE"

log_msg ">>> Step 2 Completed. "


# ==============================================================================
# Step 3: Result Filtering
# ==============================================================================
print_header "Step 3: Result Filtering "

FINAL_OUTPUT="${OUTPUT_DIR}/final_m6A_sites.txt"

# Execution 
python3 "$SCRIPT_FILTER" \
    --input "$PRED_RESULTS" \
    --threshold "$PROB_THRESHOLD" \
    --output "$FINAL_OUTPUT" \
    2>&1 | tee -a "$LOG_FILE"

log_msg ">>> Step 3 Completed. "


#!/bin/bash
# nanom6A Training Workflow Demo

TRAIN_SCRIPT="${BASE_DIR}/train.py"
FEATURES_FILE="${BASE_DIR}/train/combined_features.tsv"
MODEL_SAVE="${BASE_DIR}/retrained_models/new_nanom6a.pkl"

echo "Training nanom6A Model..."
python3 "$TRAIN_SCRIPT" \
    --input "$FEATURES_FILE" \
    --out_model "$MODEL_SAVE"