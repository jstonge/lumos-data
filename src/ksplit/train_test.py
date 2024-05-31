import argparse
from pathlib import Path
import pandas as pd
import sys

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
    current_annots = LS.get_annotations(only_annots=True)
    
    
