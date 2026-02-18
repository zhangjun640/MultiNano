import pandas as pd
import argparse
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate Read-level m6A predictions to Site-level stoichiometry."
    )
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="Input read-level prediction file (TSV format)."
    )
    parser.add_argument(
        "-o", "--output", 
        required=True, 
        help="Output site-level stoichiometry file."
    )
    parser.add_argument(
        "--min_coverage", 
        type=int, 
        default=5, 
        help="Minimum read coverage required to report a site (default: 5)."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    print(f"Reading input file: {args.input} ...")
    
    try:
        # Read TSV file
        # Assumes file contains headers: chrom, pos, alignstrand, loc_in_re, readname, strand, kmer, prob, methylation
        df = pd.read_csv(args.input, sep='\t')
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # Check if required columns exist
    required_columns = ['chrom', 'pos', 'strand', 'methylation']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Input file must contain columns: {required_columns}")
        print(f"Found columns: {list(df.columns)}")
        sys.exit(1)

    print("Aggregating reads by site...")

    # --- Core Logic ---
    # 1. Group by chromosome, position, strand, and kmer
    #    (Include kmer to preserve it; theoretically, it's the same for a specific site)
    grouped = df.groupby(['chrom', 'pos', 'strand', 'kmer'])

    # 2. Calculate statistics
    #    count: Total number of reads (Coverage)
    #    sum:   Number of modified reads (Sum of 1s in 'methylation' column)
    #    mean:  Modification rate (Mean of 'methylation' column)
    site_df = grouped['methylation'].agg(['count', 'sum', 'mean']).reset_index()

    # 3. Rename columns for clarity
    site_df.rename(columns={
        'count': 'coverage',       # Total read depth
        'sum': 'mod_count',        # Count of modified reads
        'mean': 'mod_rate'         # Modification rate (Stoichiometry)
    }, inplace=True)

    # 4. Filter sites with low coverage (Optional)
    initial_sites = len(site_df)
    site_df = site_df[site_df['coverage'] >= args.min_coverage]
    filtered_sites = len(site_df)
    
    print(f"Filtered {initial_sites - filtered_sites} sites with coverage < {args.min_coverage}")

    # 5. Format modification rate (round to 4 decimal places)
    site_df['mod_rate'] = site_df['mod_rate'].round(4)

    # 6. Save results
    print(f"Writing results to: {args.output}")
    site_df.to_csv(args.output, sep='\t', index=False)
    
    print("Done.")

if __name__ == "__main__":
    main()
