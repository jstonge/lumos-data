import argparse
from pathlib import Path
import pandas as pd
import sys
from datasets import Dataset

sys.path.append("..")
from label_studio import labelStudio

def parse_args():
    parser = argparse.ArgumentParser("Data Downloader")
    parser.add_argument(
        "-o", "--output", type=Path, help="output directory", required=True
    )
    return parser.parse_args()

def main():
    LS = labelStudio()
    current_annots = LS.get_annotations(only_annots=True, only_consensus=True)
    
    df = pd.DataFrame(current_annots)

    # extract text column from data 
    df['text'] = df.data.map(lambda x: x["text"])
    
    # why in some cases choices is not a list??
    df['sentiment'] = df.annotations.map(lambda x: x[0]['result'][0]['value']['choices'])
    df['sentiment'] = df.sentiment.map(lambda x: x[0] if isinstance(x, list) else x)
    
    # tidy data
    df = df[~df['sentiment'].str.contains('nan')]
    df = df[df['sentiment'] != 'maybe']
    df = df.drop_duplicates(subset='text')

    df['sentiment_encoded'] = df.sentiment.map({'yes': 1, 'no': 0})
    
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    ds = Dataset.from_pandas(df[['text', 'sentiment_encoded']]).train_test_split(test_size=0.2, seed=42)
    ds.push_to_hub(f"jstonge1/data-statements-{today}")
    ds.save_to_disk(f"../../data/annots/data-statements-{today}")

if __name__ == "__main__":
    main()
    
    
    
    
