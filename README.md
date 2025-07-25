# MultiNano: A Deep Learning Framework for m6A Prediction

**MultiNano** is a deep learning framework designed for predicting m6A RNA modifications using raw electrical signals from Oxford Nanopore sequencing. It provides high accuracy across species and conditions, offering a user-friendly pipeline for researchers. MultiNano supports both training from scratch and direct prediction modes, detailed as follows.

-----

## 1. Environment Setup

First, create the Conda environment using the provided `MultiNano.yml` file. This will install all necessary dependencies.

```bash
conda env create -f demo/MultiNano.yml
```

| Option              | Purpose                                                                                                 |
| :------------------ | :------------------------------------------------------------------------------------------------------- |
| `conda env create`  | Creates a new Conda environment.                                                                         |
| `-f <FILE>`         | Specifies the path to the `.yml` file that lists all required packages for the environment.              |

**Tip:** After creation, activate the new environment with `conda activate MultiNano` (the environment name is defined within the `.yml` file).

-----

## Data Pre-processing Pipeline

This section details the steps required to convert raw Nanopore data into a feature format suitable for model training or prediction.

### 2. Convert *multi-fast5* to *single-fast5*

The first step is to convert the default `multi-fast5` files from the sequencer into `single-fast5` format, as required by Tombo.

```bash
multi_to_single_fast5   -i demo/input_fast5   -s demo/output_fast5_single   --recursive   -t 30
```

| Option        | Purpose                                        |
| :------------ | :--------------------------------------------- |
| `-i <DIR>`    | Input directory containing multi-fast5 files.  |
| `-s <DIR>`    | Output directory to save single-fast5 files.   |
| `--recursive` | Recursively searches for files in subdirectories. |
| `-t <INT>`    | Number of parallel threads to use for conversion. |

### 3. Base-calling with Guppy

Next, perform base-calling on the single-fast5 files. This step generates base calls and writes them back into the fast5 files, which is essential for the subsequent re-squiggling step.

```bash
guppy_basecaller   -i /input/single_fast5   -s /output/basecall   --num_callers 30   --recursive   --fast5_out   --config rna_r9.4.1_70bps_hac.cfg
```

| Option                 | Purpose                                                                     |
| :--------------------- | :-------------------------------------------------------------------------- |
| `-i <DIR>`             | Input directory of single-fast5 files.                                      |
| `-s <DIR>`             | Output directory for FASTQ files and the updated fast5 files.               |
| `--num_callers <INT>`  | Number of parallel base-calling threads to use.                             |
| `--recursive`          | Traverses input subdirectories to find all fast5 files.                     |
| `--fast5_out`          | **Crucial flag.** Ensures base-calling results are written back to fast5 files. |
| `--config <FILE>`      | Specifies the configuration for high-accuracy RNA base-calling.             |

### 4. Re-squiggle with Tombo

Tombo's `resquiggle` command aligns the raw electronic signal events to a reference genome or transcriptome. This step is critical for accurately mapping signals to specific genomic positions.

```bash
tombo resquiggle   /path/to/workspace   /path/to/reference.fa   --rna   --overwrite   --processes 50   --corrected-group RawGenomeCorrected_001   --basecall-group Basecall_1D_001
```

| Option                    | Purpose                                                                                                                              |
| :------------------------ | :----------------------------------------------------------------------------------------------------------------------------------- |
| **Positional 1**          | Workspace directory containing the base-called single-fast5 files from the previous step.                                            |
| **Positional 2**          | The reference genome or transcriptome in FASTA format.                                                                               |
| `--rna`                   | Informs Tombo to use RNA-specific models and expectations for signal alignment.                                                      |
| `--overwrite`             | If the command was run before, this will overwrite the previous Tombo output within the fast5 files.                                |
| `--processes <INT>`       | Number of worker processes for parallel execution.                                                                                   |
| `--corrected-group <STR>` | **Important.** The name of the group within the fast5 file where Tombo will store the re-squiggled alignment data. This name is needed for feature extraction. |
| `--basecall-group <STR>`  | The name of the group within the fast5 file where Guppy stored its base calls. This must match the output from Guppy. `Basecall_1D_001` is a common default. |

### 5. Map Reads and Generate Error Profiles

This series of commands aligns the base-called reads to the reference, filters them, and generates a detailed TSV file that describes matches, mismatches, and indels for each read.

```bash
# 1. Concatenate all FASTQ files into one
cat /path/to/fast5_guppy/*.fastq > test.fastq

# 2. Align reads, convert to BAM, and sort
minimap2 -t 30 -ax map-ont ref.transcript.fa test.fastq |   samtools view -hSb |   samtools sort -@ 30 -o test.bam

# 3. Index the BAM file
samtools index test.bam

# 4. Generate a detailed error profile in TSV format
samtools view -h -F 3844 test.bam |   java -jar sam2tsv.jar -r ref.transcript.fa > test.tsv
```

* **`minimap2`**: A fast aligner optimized for noisy long-read data. `-ax map-ont` is a preset for mapping Oxford Nanopore reads.  
* **`samtools view -F 3844`**: This filter removes unmapped, secondary, supplementary, and low-quality alignments, ensuring only primary alignments are used.  
* **`sam2tsv.jar`**: A tool that converts SAM/BAM format to a detailed TSV, which is used in the next step to guide feature extraction.

### 6. Split Error Profiles for Parallel Processing

This `awk` command splits the master `test.tsv` file into smaller files, one for each read. This allows for massive parallelization in the feature extraction step.

```bash
mkdir tmp
awk 'NR==1{ h=$0 } NR>1 {
  print (!a[$2]++ ? h ORS $0 : $0) > "tmp/"$1".txt"
}' test.tsv
```

* **How it works**: The script reads `test.tsv` line by line. It saves the header (`h=$0`). For each data line, it writes the header and the current line to a new file named after the read ID (`tmp/$1.txt`). The `!a[$2]++` logic ensures the header is only written once per file.

### 7. Feature Extraction

This is the core script that extracts signal features, k-mers, and quality information for each potential m6A site (defined by the DRACH motif).

```bash
python scripts/extract.py   -i /path/to/workspace   -o test/features/   --errors_dir test/tmp/   --corrected_group RawGenomeCorrected_001   --w_is_dir yes   -k 5   -s 65   -n 30
```

| Option                   | Purpose                                                                                                                                    |
| :----------------------- | :----------------------------------------------------------------------------------------------------------------------------------------- |
| `-i <DIR>`               | Path to the directory containing the re-squiggled fast5 files.                                                                             |
| `-o <PATH>`              | Output path. Since `--w_is_dir` is set, this will be a directory where feature files are saved.                                            |
| `--errors_dir <DIR>`     | Path to the directory containing the per-read error profiles (`.txt` files) created in the previous step.                                 |
| `--corrected_group <STR>`| **Must match** the group name used in the `tombo resquiggle` command (`RawGenomeCorrected_001`). This tells the script where to find the signal data. |
| `--w_is_dir <yes/no>`    | If `yes`, treats the output path as a directory and saves features in batches to multiple files, which is efficient for large datasets.     |
| `-k <INT>`               | **K-mer length**. The length of the nucleotide sequence to extract around a site (e.g., 5 for `NN[DRACH]NN`). Must be an odd number.       |
| `-s <INT>`               | **Signal length**. The number of raw signal values to extract for each base. The script will pad or sample the signals to meet this fixed length. |
| `-n <INT>`               | Number of parallel processes to use for feature extraction.                                                                                |

### 8. Aggregate Read-Level to Site-Level Features

The final pre-processing step aggregates the features from all individual reads into a single entry for each genomic site.

```bash
# 1. Sort all feature files by genomic location (chromosome, position)
sort -k1,1 -k2,2n -k5,5 -T./ --parallel=30 -S30G test/features/* > test.features.sort.tsv &

# 2. Aggregate the sorted features
python scripts/aggre_features_to_site_level.py   --input test.features.sort.tsv   --output test.features.sort.aggre.tsv &

# 3. Filter for sites with a minimum read coverage of 20
awk 'BEGIN{ FS=OFS="	" } $4>=20 {print $0}' test.features.sort.aggre.tsv > test.features.sort.aggre.c20.tsv
```

* **`sort`**: Sorting is essential to group all reads covering the same site together. `-k1,1 -k2,2n` sorts by chromosome and then numerically by position.  
* **`aggre_features_to_site_level.py`**: This script iterates through the sorted file and combines all feature lines corresponding to the same site into a single, aggregated line.  
* **`awk '$4>=20'`**: A common quality control step to ensure that predictions are only made on sites with sufficient evidence (in this case, at least 20 reads).

-----

## Prediction Workflow

MultiNano offers several modes for prediction depending on your needs.

### 9. Site-Level Prediction

This is the standard mode for predicting m6A status directly from aggregated site-level feature files.

```bash
python scripts/predict_sl.py     --model /path/to/your/trained_site_model.pt     --input_file /path/to/your/site_features_for_prediction.txt     --output_file /path/to/your/site_predictions.tsv
```

### 10. Read-Level Prediction

If you need to determine the modification status of individual reads, use this mode.

```bash
python scripts/predict_rl.py     --model /path/to/your/trained_read_model.pt     --input_file /path/to/your/read_features_for_prediction.txt     --output_file /path/to/your/read_predictions.tsv
```

### 11. Two-Stage Prediction (Site-level with False Positive Control)

This advanced workflow first generates per-read probabilities and then uses a second model to reduce false positives. This is the recommended approach for the highest accuracy.

**Step 1: Generate Primary Predictions**

This script uses a **read-level model** to generate per-read probabilities for each site and then calculates a preliminary site-level score. The output file contains all the per-read probabilities, which are needed for the next step.

```bash
python scripts/predict_sl_with_rl.py     --model /path/to/your/trained_read_model.pt     --input_file /path/to/your/site_features_for_prediction.txt     --output_file /path/to/primary_site_predictions.csv
```

| Option          | Purpose                                                                                    |
| :-------------- | :----------------------------------------------------------------------------------------- |
| `--model`       | Path to a **read-level** trained model (`.pt`).                                            |
| `--input_file`  | Aggregated site-level feature file from step 8.                                            |
| `--output_file` | Output `.csv` file containing the site ID, all per-read probabilities, and a primary prediction. |

**Step 2: Refine Predictions and Control False Positives**

This script uses a trained XGBoost model to analyze the *distribution* of read probabilities from the previous step. It acts as a second-stage filter to differentiate true positives from likely false positives.

```bash
python FP_control/predict.py     --differentiator-model /path/to/xgboost_differentiator.json     --primary-predictions /path/to/primary_site_predictions.csv     --output /path/to/final_refined_predictions.tsv
```

| Option                   | Purpose                                                                                                                         |
| :----------------------- | :------------------------------------------------------------------------------------------------------------------------------ |
| `--differentiator-model` | Path to a pre-trained XGBoost model (`.json`) designed to identify false positive signatures.                                    |
| `--primary-predictions`  | The output file from the previous `predict_sl_with_rl.py` step, containing the per-read probability distributions.              |
| `--output`               | The final, refined prediction file containing site identifiers and a final binary prediction (1 for modified, 0 for unmodified). |

-----

## Training Workflow

You can also train MultiNano's models from scratch using your own labeled data.

### 13. Train a Site-Level Model

This trains the model that predicts modification status directly from aggregated site-level features.

```bash
python scripts/train.py     --train_file /path/to/your/site_train_data.txt     --valid_file /path/to/your/site_val_data.txt     --save_dir results/site_model_checkpoints/     --model_type comb     --epochs 100     --batch_size 64
```

| Option         | Purpose                                                                                     |
| :------------- | :------------------------------------------------------------------------------------------ |
| `--train_file` | Path to the training dataset (aggregated site-level features).                               |
| `--valid_file` | Path to the validation dataset used for monitoring training and early stopping.             |
| `--save_dir`   | Directory where model checkpoints will be saved after each epoch.                           |
| `--model_type` | Specifies the input features to use: `raw_signals`, `basecall` (k-mer), or `comb` (both).   |

### 14. Train a Read-Level Model

This trains the model that predicts modification status for each individual read. This model is required for the two-stage prediction workflow.

```bash
python scripts/train_rl.py     --train_file /path/to/your/read_train_data.txt     --valid_file /path/to/your/read_val_data.txt     --save_dir results/read_model_checkpoints/     --model_type comb     --epochs 100     --batch_size 256
```

### 15. Train a False Positive Differentiator Model

This trains the second-stage XGBoost classifier used to reduce false positives. To train it, you need primary prediction results (from `predict_sl_with_rl.py`) on sites that you **know** are true positives and true negatives.

```bash
python FP_control/train.py     --positive-samples /path/to/primary_preds_on_true_positives.csv     --negative-samples /path/to/primary_preds_on_true_negatives.csv     --model-out /path/to/save/differentiator.json
```

| Option               | Purpose                                                                                                                                             |
| :------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--positive-samples` | Path to the primary prediction results file generated by running Step 11 on a dataset where all sites are **known to be modified** (label=1).         |
| `--negative-samples` | Path to the primary prediction results file generated by running Step 11 on a dataset where all sites are **known to be unmodified** (label=0).       |
| `--model-out`        | Output path where the trained XGBoost model will be saved in `.json` format.                                                                        |
