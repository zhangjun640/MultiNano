import argparse
import logging
import os
import csv
import pandas as pd
import numpy as np
import xgboost as xgb
import scipy.stats
from sklearn.utils import resample
from statsmodels.nonparametric.smoothers_lowess import lowess

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default Model Parameters
DEFAULT_XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.1,
    'max_depth': 12,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42,
    'use_label_encoder': False
}


# Feature Extraction Function
def extract_features(probs: list) -> list:
    """
    Extracts statistical features from a list of read-level probabilities for a site.
    """
    if not probs:
        return [0] * 29

    probs = np.array(probs)
    frac_high = sum(p > 0.5 for p in probs) / len(probs)
    frac_low = sum(p < 0.05 for p in probs) / len(probs)

    feats = [
        np.max(probs), np.min(probs), np.mean(probs), np.std(probs),
        np.median(probs), np.percentile(probs, 75), np.percentile(probs, 25),
        scipy.stats.skew(probs), scipy.stats.kurtosis(probs),
        sum(p > 0.25 for p in probs) / len(probs),
        sum(p > 0.3 for p in probs) / len(probs),
        sum(p < 0.01 for p in probs) / len(probs),
        sum(p < 0.1 for p in probs) / len(probs),
        frac_high, frac_low
    ]

    for seg in np.array_split(probs, 3):
        if len(seg) > 0:
            feats.extend([
                np.mean(seg),
                np.std(seg),
                sum(p > 0.5 for p in seg) / len(seg)
            ])
        else:
            feats.extend([0, 0, 0])

    hist, _ = np.histogram(probs, bins=20, range=(0, 1), density=True)
    entropy_val = scipy.stats.entropy(hist + 1e-8)
    feats.append(entropy_val)
    feats.append(np.max(probs) - np.min(probs))
    feats.append(np.percentile(probs, 97.5) - np.percentile(probs, 2.5))

    if len(probs) > 1:
        x = np.arange(len(probs))
        slope, _, r_value, _, _ = scipy.stats.linregress(x, probs)
        feats.append(slope)
        feats.append(r_value)
    else:
        feats.extend([0, 0])

    return feats


# Data Loading, Filtering, and Feature Building Function
def load_filter_and_build_features(file_path: str) -> pd.DataFrame:
    """
    Loads data from the input file, automatically detecting two different formats.
    This function filters for rows with a predicted label of 1 and builds features for them.
    """
    logging.info(f"Loading data from {file_path} (filtering for predicted_label=1)...")
    features_list = []

    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            read_probs = []

            try:
                # Core Change: Automatic format detection and parsing
                # Format A: Nested format (id \t pos \t 'data,string')
                if len(row) == 3 and ',' in row[2]:
                    data_string = row[2]
                    fields = data_string.split(',')
                    if len(fields) >= 3 and int(fields[-1]) == 1:
                        read_probs = [float(x) for x in fields[1:-2] if x.strip()]

                # Format B: Flat format ("id\tpos\tstrand" \t prob1 \t ...)
                elif len(row) > 3 and int(row[-1]) == 1:
                    read_probs = [float(x) for x in row[1:-1] if x.strip()]

                # If neither format is matched, or predicted label is not 1, skip silently
                else:
                    continue

                # --- Feature Extraction ---
                if read_probs:
                    features = extract_features(read_probs)
                    features_list.append(features)

            except (ValueError, IndexError) as e:
                logging.warning(f"Error processing row, skipping: {row}. Error: {e}")
                continue

    if not features_list:
        return pd.DataFrame()

    feature_names = [
        'max', 'min', 'mean', 'std', 'median', 'q75', 'q25', 'skew', 'kurt',
        'frac_gt_025', 'frac_gt_03', 'frac_lt_001', 'frac_lt_01',
        'frac_gt_05', 'frac_lt_005',
        'seg1_mean', 'seg1_std', 'seg1_frac',
        'seg2_mean', 'seg2_std', 'seg2_frac',
        'seg3_mean', 'seg3_std', 'seg3_frac',
        'entropy', 'value_range', 'ci_width', 'slope', 'r_value'
    ]

    actual_feature_count = len(features_list[0])
    if len(feature_names) != actual_feature_count:
        logging.warning(f"Feature name count ({len(feature_names)}) does not match actual feature count ({actual_feature_count}). Using generic names.")
        feature_names = [f'feat_{i}' for i in range(actual_feature_count)]

    return pd.DataFrame(features_list, columns=feature_names)


# Main Training Workflow
def run_differentiator_training(positive_labeled_path: str, negative_labeled_path: str, model_save_path: str):
    """
    Executes the complete training workflow for the false positive differentiator model,
    including internal filtering logic.
    """
    logging.info("Starting the false positive differentiator model training pipeline...")

    x_tp = load_filter_and_build_features(positive_labeled_path)
    if x_tp.empty:
        logging.error(f"No samples with predicted_label=1 found in {positive_labeled_path} (TPs). Aborting training.")
        return
    y_tp = pd.Series([1] * len(x_tp), name="is_true_positive")
    logging.info(f"Successfully filtered and built features for {len(x_tp)} True Positive (TP) samples.")

    x_fp = load_filter_and_build_features(negative_labeled_path)
    if x_fp.empty:
        logging.error(f"No samples with predicted_label=1 found in {negative_labeled_path} (FPs). Aborting training.")
        return
    y_fp = pd.Series([0] * len(x_fp), name="is_true_positive")
    logging.info(f"Successfully filtered and built features for {len(x_fp)} False Positive (FP) samples.")

    x_train_raw = pd.concat([x_tp, x_fp], ignore_index=True)
    y_train_raw = pd.concat([y_tp, y_fp], ignore_index=True)

    logging.info("Performing data balancing (upsampling)...")
    training_data = pd.concat([x_train_raw, y_train_raw], axis=1)

    tp_samples = training_data[training_data.is_true_positive == 1]
    fp_samples = training_data[training_data.is_true_positive == 0]

    if len(tp_samples) == 0 or len(fp_samples) == 0:
        logging.error("One of the classes (TP or FP) is empty. Cannot proceed with training.")
        return

    if len(tp_samples) > len(fp_samples):
        majority_class, minority_class = tp_samples, fp_samples
    else:
        majority_class, minority_class = fp_samples, tp_samples

    minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
    balanced_data = pd.concat([majority_class, minority_upsampled])

    logging.info(
        f"After upsampling, TP samples: {len(balanced_data[balanced_data.is_true_positive == 1])}, FP samples: {len(balanced_data[balanced_data.is_true_positive == 0])}")

    x_train_balanced = balanced_data.drop("is_true_positive", axis=1)
    y_train_balanced = balanced_data["is_true_positive"]

    logging.info("Training XGBoost differentiator model...")
    model = xgb.XGBClassifier(**DEFAULT_XGB_PARAMS)
    model.fit(x_train_balanced, y_train_balanced)

    model_dir = os.path.dirname(model_save_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save_model(model_save_path)
    logging.info(f"✅ Model successfully trained and saved to: {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an XGBoost model to differentiate between True Positives (TP) and False Positives (FP). The script automatically filters for samples with a predicted label of 1.")
    parser.add_argument("--positive-samples",
                        type=str, required=True,
                        help="Path to the prediction results file for all samples with a true label of 1.")
    parser.add_argument("--negative-samples",
                        type=str, required=True,
                        help="Path to the prediction results file for all samples with a true label of 0.")
    parser.add_argument("--model-out",
                         type=str,
                        required=True,
                        help="Output path to save the trained differentiator model (e.g., 'fp_differentiator_model.json').")

    args = parser.parse_args()

    run_differentiator_training(
        positive_labeled_path=args.positive_samples,
        negative_labeled_path=args.negative_samples,
        model_save_path=args.model_out
    )