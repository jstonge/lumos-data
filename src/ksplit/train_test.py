import argparse
from pathlib import Path
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser("Data Downloader")
    parser.add_argument(
        "-i", "--input", type=Path, help="input directory", required=True
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="output directory", required=True
    )
    return parser.parse_args()



def main():
    
    # round2_merged.to_parquet(output_dir / "training_data_2023-12-07.parquet")


    # collective_truth = pd.read_csv("../output/collective_truth.csv")

    # collective_truth = collective_truth[['text', 'collective_score', 'corpusid']]
    # collective_truth.rename(columns={'collective_score': 'sentiment'}, inplace=True)
    # collective_truth['sentiment'] = np.where(collective_truth.sentiment == 1, 'yes', 'no')
    
    # collective_truth.to_parquet(output_dir / "test_data_2023-12-07.parquet")
    

