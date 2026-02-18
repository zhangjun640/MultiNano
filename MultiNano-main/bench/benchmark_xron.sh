#!/bin/bash

# ==============================================================================
# Xron m6A Detection Workflow Demo
# ==============================================================================
# Description: 
#   This script demonstrates the full workflow of Xron for m6A detection.
#   Xron is a CNN+RNN+CTC basecaller that identifies m6A from raw signals.
# ==============================================================================

# --- Configuration (Environment & Paths) ---

# 1. Base Setup 
BASE_DIR=$(pwd)
OUTPUT_DIR="${BASE_DIR}/xron_demo_output"
mkdir -p "$OUTPUT_DIR"

XRON_CMD="xron"
SCRIPT_EXTRACT="${BASE_DIR}/xron-samples/extract_m6a_from_bam.py"


MODEL_NAME="models/ENEYFT"

# Input Raw Data (Folder containing .fast5 or .pod5 files)
INPUT_DATA="${BASE_DIR}/test_data/raw_fast5"

# Reference Genome (Fasta file)
REFERENCE_FASTA="${BASE_DIR}/test_data/reference.fasta"

THREADS=16         # CPU threads 
GPU_DEVICE=0       # Target GPU ID 

# Log file 
LOG_FILE="${OUTPUT_DIR}/xron_execution.log"

> "$LOG_FILE"

# ==============================================================================
# Step 1: Model Initialization
# ==============================================================================

# Execution 
$XRON_CMD init 2>&1 | tee -a "$LOG_FILE"

log_msg ">>> Step 1 Completed."


# ==============================================================================
# Step 2: m6A-Aware Basecalling
# ==============================================================================

BASECALL_OUT="${OUTPUT_DIR}/xron_calls"
export CUDA_VISIBLE_DEVICES=$GPU_DEVICE

$XRON_CMD call \
    -i "$INPUT_DATA" \
    -o "$BASECALL_OUT" \
    -m "$MODEL_NAME" \
    --fast5 \
    --beam 50 \
    --chunk_len 2000 \
    2>&1 | tee -a "$LOG_FILE"

log_msg ">>> Step 2 Completed. "


# ==============================================================================
# Step 3: Site-Level m6A Extraction
# ==============================================================================

# Locate the tagged BAM file (assumed output from Step 2)
INPUT_BAM=$(find "$BASECALL_OUT" -name "*.bam" | head -n 1)

SITE_SUMMARY_OUT="${OUTPUT_DIR}/m6A_site_summary"
mkdir -p "$SITE_SUMMARY_OUT"

python3 "$SCRIPT_EXTRACT" \
    --bam "$INPUT_BAM" \
    --ref "$REFERENCE_FASTA" \
    --out "$SITE_SUMMARY_OUT" \
    --motif "DRACH" \
    2>&1 | tee -a "$LOG_FILE"

log_msg ">>> Step 3 Completed. "

