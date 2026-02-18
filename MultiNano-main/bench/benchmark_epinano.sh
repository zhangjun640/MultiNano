#!/bin/bash

# ==============================================================================
# EpiNano m6A Detection Workflow Demo
# ==============================================================================
# Description: 
#   This script demonstrates the full workflow of EpiNano for m6A detection.
#   It processes data from BAM files to SVM-based modification predictions.
# ==============================================================================

# --- Configuration (Environment & Paths) ---

# 1. Base Setup 
BASE_DIR=$(pwd)
OUTPUT_DIR="${BASE_DIR}/epinano_demo_output"
mkdir -p "$OUTPUT_DIR"

# 2. Script Paths
SCRIPT_VARIANTS="${BASE_DIR}/Epinano_Variants.py"
SCRIPT_SLIDE="${BASE_DIR}/misc/Slide_Variants.py"
SCRIPT_PREDICT="${BASE_DIR}/Epinano_Predict.py"

# 3. Model and Input Files
# Path to the pre-trained m6A SVM model (typically in the models/ folder)
MODEL_FILE="${BASE_DIR}/models/rrach.6.q3.mis3.del3.MODEL.linear.model.dump"

# Input files: Indexed BAM and Reference FASTA
INPUT_BAM="${BASE_DIR}/test_data/sample.bam"
REFERENCE_FASTA="${BASE_DIR}/test_data/reference.fasta"

# 4. Runtime Parameters
THREADS=16         # CPU threads

# Log file
LOG_FILE="${OUTPUT_DIR}/epinano_execution.log"

# Clear previous logs
> "$LOG_FILE"


# ==============================================================================
# Step 1: Extract Base-calling Error Features
# ==============================================================================
print_header "Step 1: Feature Extraction"
log_msg "Description: Extracting mis-matches, indels, and quality scores per site."


python3 "$SCRIPT_VARIANTS" \
    -b "$INPUT_BAM" \
    -r "$REFERENCE_FASTA" \
    -c "$THREADS" \
    -o "$OUTPUT_DIR" \
    2>&1 | tee -a "$LOG_FILE"

VAR_OUTPUT=$(find "$OUTPUT_DIR" -name "*.per.site.var.csv" | head -n 1)

log_msg ">>> Step 1 Completed."


# ==============================================================================
# Step 2: Slide Variants (Generate k-mer windows)
# ==============================================================================

SLIDE_OUTPUT="${OUTPUT_DIR}/sample.per_site.5mer.csv"

# Execution
python3 "$SCRIPT_SLIDE" \
    -i "$VAR_OUTPUT" \
    -o "$SLIDE_OUTPUT" \
    2>&1 | tee -a "$LOG_FILE"

log_msg ">>> Step 2 Completed."


# ==============================================================================
# Step 3: m6A Prediction using SVM
# ==============================================================================
print_header "Step 3: SVM Prediction "
log_msg "Description: Predicting m6A sites using the pre-trained SVM model."

PRED_PREFIX="${OUTPUT_DIR}/m6A_predictions"


python3 "$SCRIPT_PREDICT" \
    --model "$MODEL_FILE" \
    --predict "$SLIDE_OUTPUT" \
    --columns 8,13,23 \
    --out_prefix "$PRED_PREFIX" \
    2>&1 | tee -a "$LOG_FILE"

#!/bin/bash
# EpiNano Training Workflow Demo

TRAIN_SCRIPT="${BASE_DIR}/Epinano_Predict.py"
COMBINED_CSV="${BASE_DIR}/train/ko_wt_combined.5mer.csv"
COLUMNS="8,13,23" 
LABEL_COL="26"   
OUTPUT_PREFIX="retrained_epinano_svm"
python3 "$TRAIN_SCRIPT" \
    --train "$COMBINED_CSV" \
    --predict "$COMBINED_CSV" \
    --columns "$COLUMNS" \
    --modification_status_column "$LABEL_COL" \
    --accuracy_estimation \
    --kernel linear \
    --out_prefix "$OUTPUT_PREFIX"