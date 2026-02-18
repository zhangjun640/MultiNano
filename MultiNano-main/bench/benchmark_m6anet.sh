#!/bin/bash

# ==============================================================================
# m6Anet Detection Workflow Demo
# ==============================================================================
# Description: 
#   This script demonstrates the full workflow of m6Anet for m6A detection.
#   It processes Nanopolish/f5c eventalign output into m6A probability scores.
# ==============================================================================

# --- Configuration (Environment & Paths) ---

# 1. Base Setup 
BASE_DIR=$(pwd)
OUTPUT_DIR="${BASE_DIR}/m6anet_demo_output"
mkdir -p "$OUTPUT_DIR"


INPUT_EVENTALIGN="${BASE_DIR}/test_data/eventalign.txt"

THREADS=16        

DEVICE="cuda"      

LOG_FILE="${OUTPUT_DIR}/m6anet_execution.log"

> "$LOG_FILE"


# ==============================================================================
# Step 1: Data Preprocessing (Dataprep)
# ==============================================================================
print_header "Step 1: Data Preprocessing "

PREPRE_OUT_DIR="${OUTPUT_DIR}/dataprep_results"

# Execution 
m6anet dataprep \
    --eventalign "$INPUT_EVENTALIGN" \
    --out_dir "$PREPRE_OUT_DIR" \
    --n_processes "$THREADS" \
    2>&1 | tee -a "$LOG_FILE"

log_msg ">>> Step 1 Completed. "


# ==============================================================================
# Step 2: m6A Inference
# ==============================================================================
print_header "Step 2: m6A Inference"

INFERENCE_OUT_DIR="${OUTPUT_DIR}/inference_results"

# Execution 
m6anet inference \
    --input_dir "$PREPRE_OUT_DIR" \
    --out_dir "$INFERENCE_OUT_DIR" \
    --n_processes "$THREADS" \
    --device "$DEVICE" \
    2>&1 | tee -a "$LOG_FILE"

log_msg ">>> Step 2 Completed."



TRAIN_DATA_DIR="${BASE_DIR}/processed_train_data"
OUTPUT_DIR="retrained_m6anet"

echo "Starting m6Anet Training..."
m6anet train \
    --input_dir "$TRAIN_DATA_DIR" \
    --out_dir "$OUTPUT_DIR" \
    --epochs 10 \
    --device cuda