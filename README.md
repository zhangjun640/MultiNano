# MultiNano

MultiNano is a deep learning framework designed for predicting m6A RNA modifications using raw electrical signals from Oxford Nanopore sequencing. It provides high accuracy across species and conditions, offering a user-friendly pipeline for researchers. MultiNano supports both training from scratch and direct prediction modes, detailed as follows.

---

## 1  Environment Setup

```bash
conda env create -f demo/environment.yml
```

| Option             | Purpose                                                           |
| ------------------ | ----------------------------------------------------------------- |
| `conda env create` | Creates a new Conda environment.                                  |
| `-f <FILE>`        | Path to the `environment.yml` file listing all required packages. |

**Tip:** Activate the environment afterwards with `conda activate MultiNano` (or use the environment name defined in `environment.yml`).

---

## 2  Convert *multi-fast5* to *single-fast5*

```bash
multi_to_single_fast5 \
  -i demo/input_fast5 \
  -s demo/output_fast5_single \
  --recursive \
  -t 30
```

| Option        | Purpose                                  |
| ------------- | ---------------------------------------- |
| `-i <DIR>`    | Input directory with multi-fast5 files.  |
| `-s <DIR>`    | Output directory for single-fast5 files. |
| `--recursive` | Recursively search subdirectories.       |
| `-t <INT>`    | Number of parallel threads.              |

---

## 3  Base-calling with *Guppy*

```bash
guppy_basecaller \
  -i /input/single_fast5 \
  -s /output/basecall \
  --num_callers 30 \
  --recursive \
  --fast5_out \
  --config rna_r9.4.1_70bps_hac.cfg
```

| Option                | Purpose                                             |
| --------------------- | --------------------------------------------------- |
| `-i <DIR>`            | Input directory of single-fast5 files.              |
| `-s <DIR>`            | Output directory for FASTQ and updated fast5 files. |
| `--num_callers <INT>` | Number of base-calling threads.                     |
| `--recursive`         | Traverse subdirectories.                            |
| `--fast5_out`         | Write modified fast5 files.                         |
| `--config <FILE>`     | Configuration for RNA high-accuracy base-calling.   |

---

## 4  Re-squiggle with *Tombo*

```bash
tombo resquiggle \
  /path/to/workspace \
  /path/to/reference.fa \
  --rna \
  --overwrite \
  --processes 50 \
  --fit-global-scale \
  --include-event-stdev \
  --corrected-group RawGenomeCorrected_000 \
  --basecall-group Basecall_1D_001 \
  > resquiggle.log 2>&1
```

| Option                    | Purpose                                         |
| ------------------------- | ----------------------------------------------- |
| Positional 1              | Workspace directory with base-called fast5s.    |
| Positional 2              | Reference genome/transcriptome in FASTA format. |
| `--rna`                   | Use RNA model expectations.                     |
| `--overwrite`             | Overwrite Tombo output.                         |
| `--processes <INT>`       | Number of worker processes.                     |
| `--fit-global-scale`      | Apply global scaling.                           |
| `--include-event-stdev`   | Save event standard deviations.                 |
| `--corrected-group <STR>` | Group name for corrected signals.               |
| `--basecall-group <STR>`  | Base-call group name in fast5.                  |

---

## 5  Map Reads to the Reference Transcriptome

```bash
cat /path/to/fast5_guppy/*.fastq > test.fastq

minimap2 -t 30 -ax map-ont ref.transcript.fa test.fastq | \
  samtools view -hSb | \
  samtools sort -@ 30 -o test.bam

samtools index test.bam

samtools view -h -F 3844 test.bam | \
  java -jar sam2tsv.jar -r ref.transcript.fa > test.tsv
```

---

## 6  Split `test.tsv` for Parallel Feature Extraction

```bash
mkdir tmp
awk 'NR==1{ h=$0 } NR>1 {
  print (!a[$2]++ ? h ORS $0 : $0) > "tmp/"$1".txt"
}' test.tsv
```

| Component      | Description             |
| -------------- | ----------------------- |
| `NR==1`        | Store the header.       |
| `a[$2]++`      | Deduplicate by read ID. |
| `> tmp/$1.txt` | Output file per read.   |

---

## 7  Feature Extraction

```bash
python scripts/extract_features.py \
  -i /path/to/workspace \
  -o test/features/ \
  --errors_dir test/tmp/ \
  --corrected_group RawGenomeCorrected_001 \
  -b test/reference.bed \
  --w_is_dir 1 -k 5 -s 65 -n 30
```

Each row in the output represents one sample with columns such as:

```
chrom	position	alignstrand	loc_in_ref	readname	strand	k_mer	signals	quality	mis	ins	del
```

---

## 8  Aggregate Read-level Features to Site-level

```bash
sort -k1,1 -k2,2n -k5,5 -T./ --parallel=30 -S30G test/features/* > test.features.sort.tsv &
python scripts/aggre_features_to_site_level.py \
  --input test.features.sort.tsv \
  --output test.features.sort.aggre.tsv &
awk 'BEGIN{ FS=OFS="\t" } $4>=20 {print $0}' test.features.sort.aggre.tsv > test.features.sort.aggre.c20.tsv
```

---

## 9  Predict m6A Sites

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/predict.py \
  --input_file test.features.sort.aggre.c20.tsv \
  --model models/mil_allspecies_model_states.pt
```

| Option         | Description                                 |
| -------------- | ------------------------------------------- |
| `--input_file` | Aggregated site-level features.             |
| `--model`      | Path to the pretrained or checkpoint model. |

---

## 10  Train Your Own Model

```bash
awk -v OFS="\t" '{print $0, 1}' mod_features.txt > mod_features_labeled.txt

CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --train_option 0 \
  --train_file test/train.txt \
  --valid_file test/valid.txt \
  --epochs 50 \
  --patience 5 \
  --seq_lens 5 \
  --signal_lens 65 \
  --batch_size 4 \
  --hidden_size 128 \
  --dropout_rate 0.5 \
  --clip_grad 0.5 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --embedding_size 4 \
  --num_workers 2 \
  --model_type comb_basecall_raw \
  --save_dir train/
```

---
