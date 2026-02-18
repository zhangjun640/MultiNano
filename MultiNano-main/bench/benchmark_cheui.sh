#!/bin/bash

# ==============================================================================
# CHEUI m6A Detection Workflow Demo
# ==============================================================================
# Description: 
#   This script demonstrates the full workflow of CHEUI for m6A detection.
#   It processes data step-by-step for clear visualization.
# ==============================================================================

# --- Configuration (Environment & Paths) ---

# 1. Base Setup
BASE_DIR=$(pwd)
OUTPUT_DIR="${BASE_DIR}/demo_output_m6A"
mkdir -p "$OUTPUT_DIR"

# 2. Script & Model Paths (Localized based on repository structure)
SCRIPT_PREPROCESS="${BASE_DIR}/scripts/CHEUI_preprocess_m6A.py"
SCRIPT_MODEL1="${BASE_DIR}/scripts/CHEUI_predict_model1.py"
SCRIPT_MODEL2="${BASE_DIR}/scripts/CHEUI_predict_model2.py"

MODEL1_FILE="${BASE_DIR}/CHEUI_trained_models/CHEUI_m6A_model1.h5"
MODEL2_FILE="${BASE_DIR}/CHEUI_trained_models/CHEUI_m6A_model2.h5"
KMER_MODEL="${BASE_DIR}/kmer_models/model_kmer.csv"

# 3. Input Files (Using built-in test data)
INPUT_NANOPOLISH="${BASE_DIR}/test/nanopolish_output_test.txt"

# 4. Runtime Parameters 
THREADS=16         # CPU threads 
GPU_DEVICE=0       # Target GPU ID
SAMPLE_LABEL="Demo_m6A"

# Log file
LOG_FILE="${OUTPUT_DIR}/execution.log"

# Clear previous logs
> "$LOG_FILE"

# ==============================================================================
# Execution Start
# ==============================================================================

log_msg "Starting CHEUI m6A Detection Demo..."
log_msg "Input File : $INPUT_NANOPOLISH"
log_msg "Output Dir : $OUTPUT_DIR"

# ==============================================================================
# Step 1: Preprocessing (Python Version)
# ==============================================================================
print_header "Step 1: Preprocessing (Nanopolish to Signals)"

PREPROCESS_OUT_DIR="${OUTPUT_DIR}/m6A_signals"

# Execution
python3 "$SCRIPT_PREPROCESS" \
    -i "$INPUT_NANOPOLISH" \
    -m "$KMER_MODEL" \
    -o "$PREPROCESS_OUT_DIR" \
    -n "$THREADS" \
    2>&1 | tee -a "$LOG_FILE"

log_msg ">>> Step 1 Completed."


# ==============================================================================
# Step 2: Model 1 Prediction (Read-Level)
# ==============================================================================
print_header "Step 2: Model 1 Prediction (Read-Level Probability) "

# Find the generated signal file 
SIGNAL_FILE=$(find "$PREPROCESS_OUT_DIR" -name "*signals+IDS.p" | head -n 1)

if [ -z "$SIGNAL_FILE" ]; then
    log_msg "Error: Signal file not found!"
    exit 1
fi

OUTPUT_MODEL1="${OUTPUT_DIR}/read_level_m6A.txt"
export CUDA_VISIBLE_DEVICES=$GPU_DEVICE

# Execution 
python3 "$SCRIPT_MODEL1" \
    -i "$SIGNAL_FILE" \
    -m "$MODEL1_FILE" \
    -o "$OUTPUT_MODEL1" \
    -l "$SAMPLE_LABEL" \
    2>&1 | tee -a "$LOG_FILE"

log_msg ">>> Step 2 Completed."


# ==============================================================================
# Step 3: Sorting Predictions
# ==============================================================================
print_header "Step 3: Sorting Read-Level Predictions"

OUTPUT_SORTED="${OUTPUT_DIR}/read_level_m6A_sorted.txt"

# Execution
sort -k1 --parallel="$THREADS" "$OUTPUT_MODEL1" > "$OUTPUT_SORTED" 2>> "$LOG_FILE"

log_msg ">>> Step 3 Completed."


# ==============================================================================
# Step 4: Model 2 Prediction (Site-Level)
# ==============================================================================
print_header "Step 4: Model 2 Prediction (Site-Level Stoichiometry) "
OUTPUT_MODEL2="${OUTPUT_DIR}/site_level_m6A.txt"

# Execution
python3 "$SCRIPT_MODEL2" \
    -i "$OUTPUT_SORTED" \
    -m "$MODEL2_FILE" \
    -o "$OUTPUT_MODEL2" \
    2>&1 | tee -a "$LOG_FILE"

log_msg ">>> Step 4 Completed."

