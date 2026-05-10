import argparse
import json

import pandas as pd


def summarize_group_types(csv_path):
    df = pd.read_csv(csv_path)
    return df["group_type"].value_counts(normalize=True).to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    summary = summarize_group_types(args.csv)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
