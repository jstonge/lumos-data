import pandas as pd
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser("Data Downloader")
    parser.add_argument(
        "-o", "--output", type=Path, help="output directory", required=True
    )
    return parser.parse_args()

def main():
    OUTPUT_DIR = parse_args().output
    
    url = 'https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=bus'

    business_econ_mgmt = pd.read_html(url)
    business_econ_mgmt = business_econ_mgmt[0]
    url = 'https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=chm'

    Chem_MatSci = pd.read_html(url)
    Chem_MatSci = Chem_MatSci[0]
    url = 'https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=eng'

    Eng_CS = pd.read_html(url)
    Eng_CS = Eng_CS[0]
    url = 'https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=med'

    Health_Med = pd.read_html(url)
    Health_Med = Health_Med[0]
    url = 'https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=hum'

    Hum_Lit_Art = pd.read_html(url)
    Hum_Lit_Art = Hum_Lit_Art[0]
    url = 'https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=bio'

    Life_Earth_Sci = pd.read_html(url)
    Life_Earth_Sci = Life_Earth_Sci[0]
    url = 'https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=phy'

    Phys_Math = pd.read_html(url)
    Phys_Math = Phys_Math[0]
    url = 'https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=soc'

    Social_Sci = pd.read_html(url)
    Social_Sci = Social_Sci[0]
    combined_data = pd.concat([Chem_MatSci, business_econ_mgmt, Eng_CS, Health_Med, Hum_Lit_Art, 
                            Life_Earth_Sci, Phys_Math, Social_Sci], axis=0)


    combined_data.to_csv(OUTPUT_DIR / 'journalsbyfield.csv', index=False)

if __name__ == '__main__':
    main()