import argparse
import logging
import os
import csv
import pandas as pd
import numpy as np
import xgboost as xgb
import scipy.stats

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Feature Extraction Function (Consistent with training)
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


#Main Prediction Workflow
def run_refined_prediction(differentiator_model_path: str, primary_predictions_path: str, output_path: str):
    """
    Uses a trained differentiator model to refine primary prediction results to control for false positives.
    This version can automatically detect and handle two different input file formats.
    """
    logging.info("Starting two-stage refined prediction pipeline...")

    logging.info(f"Loading differentiator model: {differentiator_model_path}")
    differentiator_model = xgb.XGBClassifier()
    differentiator_model.load_model(differentiator_model_path)

    feature_names = [
        'max', 'min', 'mean', 'std', 'median', 'q75', 'q25', 'skew', 'kurt',
        'frac_gt_025', 'frac_gt_03', 'frac_lt_001', 'frac_lt_01',
        'frac_gt_05', 'frac_lt_005',
        'seg1_mean', 'seg1_std', 'seg1_frac',
        'seg2_mean', 'seg2_std', 'seg2_frac',
        'seg3_mean', 'seg3_std', 'seg3_frac',
        'entropy', 'value_range', 'ci_width', 'slope', 'r_value'
    ]

    logging.info(f"Reading primary prediction results from: {primary_predictions_path}")
    logging.info(f"Refined results will be saved to: {output_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(primary_predictions_path, 'r') as infile, open(output_path, 'w', newline='') as outfile:
        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile, delimiter='\t')

        # Use a more generic header
        writer.writerow(['sample_info', 'refined_prediction'])

        processed_count = 0
        refined_pos_count = 0

        for row in reader:
            sample_info_str, read_probs, primary_prediction = '', [], -1

            try:
                # Core Change: Automatic format detection and parsing
                # Format A: Nested format (id \t pos \t 'data,string')
                if len(row) == 3 and ',' in row[2]:
                    sample_info_str = f"{row[0]}_{row[1]}"
                    data_string = row[2]
                    fields = data_string.split(',')
                    if len(fields) >= 3:
                        primary_prediction = int(fields[-1])
                        read_probs = [float(x) for x in fields[1:-2] if x.strip()]

                # Format B: Flat format ("id\tpos\tstrand" \t prob1 \t ...)
                elif len(row) > 3:
                    sample_info_str = row[0]
                    primary_prediction = int(row[-1])
                    read_probs = [float(x) for x in row[1:-1] if x.strip()]

                # If neither format is matched, skip the row
                else:
                    logging.warning(f"Skipping row with unknown format: {row}")
                    continue

                # --- Subsequent prediction logic remains unchanged ---
                final_prediction = 0
                if primary_prediction == 1:
                    if read_probs:
                        features = extract_features(read_probs)
                        features_df = pd.DataFrame([features], columns=feature_names)
                        final_prediction = differentiator_model.predict(features_df)[0]

                if final_prediction == 1:
                    refined_pos_count += 1

                writer.writerow([sample_info_str, final_prediction])
                processed_count += 1
                if processed_count % 10000 == 0:
                    logging.info(f"Processed {processed_count} rows...")

            except Exception as e:
                logging.warning(f"An exception occurred while processing a row, skipping: {row}. Error: {e}")

    logging.info(f"✅ Refined prediction complete! Processed a total of {processed_count} rows.")
    logging.info(f"Total number of sites predicted as positive: {refined_pos_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Refine primary prediction results using a trained false positive differentiator model. This script can automatically detect multiple input formats.")
    parser.add_argument("--differentiator-model",
                        type=str,
                        required=True,
                        help="Path to the trained false positive differentiator model.")
    parser.add_argument("--primary-predictions",type=str,
                        required=True,
                        help="Path to the primary prediction results file that needs to be refined.")
    parser.add_argument("--output", type=str,
                        required=True,
                        help="Path to save the final, refined prediction results.")

    args = parser.parse_args()

    run_refined_prediction(
        differentiator_model_path=args.differentiator_model,
        primary_predictions_path=args.primary_predictions,
        output_path=args.output
    )