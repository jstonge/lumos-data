# import jsonlines
import pandas as pd
import re
# from bson import json_util
import json
import spacy

from pathlib import Path
from tqdm import tqdm
import argparse

from pymongo import MongoClient

def parse_args():
    parser = argparse.ArgumentParser("Data Downloader")
    parser.add_argument(
        "-i", "--input", type=Path, help="input directory", required=True
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="output directory", required=True
    )
    return parser.parse_args()

def get_journals_display_name(db, df_google):
    df_oa = pd.DataFrame(list(db.venues_oa.find( {}, 
        {"display_name": 1, "alternate_titles":1, "abbreviated_title": 1, "ids": 1}
    )))
    
    oa_display_name2googlepub = {
        'Review of Financial Studies': 'the review of financial studies',
        'Applied Catalysis B-environmental': 'applied catalysis b: environmental',
        'Journal of energy & environmental sciences': 'energy & environmental science',
        'Journal of materials chemistry. A, Materials for energy and sustainability': 'journal of materials chemistry a',
        'The American Economic Review': 'american economic review',
        'Quarterly Journal of Economics': 'the quarterly journal of economics',
        'Journal of Finance': 'the journal of finance',
        'Physical Review X': 'physical review. x',
        'European Physical Journal C': 'the european physical journal c',
        'Nature Reviews Molecular Cell Biology': 'nature reviews. molecular cell biology',
        'Journal of Religion & Health': 'journal of religion and health',
        'Lancet Oncology': 'the lancet oncology',
        'Lancet Infectious Diseases': 'the lancet infectious diseases',
        'Astronomy and Astrophysics': 'astronomy & astrophysics',
        'Light-Science & Applications': 'light: science & applications',
        'Energy research and social science': 'energy research & social science',
        'Global Environmental Change-human and Policy Dimensions': 'global environmental change',
        'Journalism: Theory, Practice & Criticism': 'journalism',
    }

    # remove duplicated? Why do we have duplicated in our db.
    df_oa = df_oa[~df_oa.display_name.duplicated()]

    # get everything lower case wihtout changing display_name.
    def lower_display_name(x):
        return [oa_display_name2googlepub[x].lower() 
                if oa_display_name2googlepub.get(x) 
                else x.lower() 
                for x in x.display_name]
    
    df_oa = df_oa.assign(display_name_lower = lower_display_name)

    df_oa['issn'] = df_oa.ids.map(lambda x: x['issn'][0] if x.get('issn') else None)

    return df_google.merge(df_oa, how="left", left_on="Publication", right_on="display_name_lower")

def add_journals_via_issnl(db, metadata_venue, df_google):
    # missing venues, if anyone wants to keep looking for them in the db.
    # mostly coming from computer science journals.
    missing_venues = metadata_venue[metadata_venue.display_name.isna()]
    metadata_venue = metadata_venue[~metadata_venue.display_name.isna()]
    
    # oa_issnl2googlepub = {
    #     '1063-6919', 'ieee/cvf conference on computer vision and pattern recognition',
    #     '1364-0321', 'renewable and sustainable energy reviews'
    #     '2159-5399', 'aaai conference on artificial intelligence',
    #     '1520-6149', 'ieee international conference on acoustics, speech and signal processing (icassp)'
    # }
    
    # last one missing
    # '1550-5499', 'ieee/cvf international conference on computer vision',

    list_issnl = ['1063-6919', '1364-0321', '2159-5399', '1520-6149']
    
    df_oa_issnl = pd.DataFrame(list(db.venues_oa.find(
        {"issn_l": {"$in": list_issnl}}, 
        {"display_name": 1, "alternate_titles":1, "abbreviated_title": 1, "ids": 1, 'issn_l': 1}
    )))
    
    missing_metadata_venue = df_google.loc[(df_google.Publication.isin(missing_venues.Publication.tolist())) & (df_google.Publication != 'ieee/cvf international conference on computer vision'),:]\
             .assign(issn_l = list_issnl)\
             .merge(df_oa_issnl, how="left", on="issn_l")

    return pd.concat([metadata_venue, missing_metadata_venue], axis=0).reset_index(drop=True)

def main():
    args = parse_args()
    INPUT_DIR = args.input
    OUTPUT_DIR = args.output

    from pymongo import MongoClient

    pw = "password"
    uri = f"mongodb://cwward:{pw}@wranglerdb01a.uvm.edu:27017/?authSource=admin&readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false"
    client = MongoClient(uri)

    db = client['papersDB']
    
    df_google = pd.read_csv(INPUT_DIR / "journalsbyfield.csv", usecols=['Publication', 'h5-index', 'h5-median'])\
                  .assign(Publication = lambda x: x.Publication.str.lower())
    
    meta_f = Path(INPUT_DIR / "metadata_venue.csv")
    metadata_venue = pd.read_csv(meta_f) if meta_f.exists() else get_journals_display_name(db, df_google)
    metadata_venue = add_journals_via_issnl(db, metadata_venue, df_google)
    # metadata_venue.to_csv("metadata_venue.csv", index=False)
    
    #!TODO: take care of duplication
    metadata_venue = metadata_venue[~metadata_venue.display_name.duplicated()]
    
    # dropping nas. 170/180 left.
    # metadata_venue = metadata_venue[~metadata_venue.display_name.isna()]

    # Grab a random sample of metadata about papers from mongoDB.
    # The issue here is that we do not know the field of study just by looking at papers,
    # so we need to identify the corpusIDs given the field first. 

    # We are using display name instead of issn because few journals/conferene do not seem to have
    # issn. At this point, display name should work fine though.
    tot_venue_set = set(metadata_venue.display_name)
    
    nlp = spacy.load("en_core_web_trf", enable=["tok2vec"])

    i = 0 
    for venue in tot_venue_set:
        # venue = 'Applied Linguistics'
        # venue = list(tot_venue_set)[0]        
        print(f"doing {venue} (done {i}/{len(tot_venue_set)})")
        venue_df = pd.DataFrame(list(db.papers.aggregate([
            { "$match": { 
                'works_oa.host_venue.display_name': venue, 
                's2orc_parsed': True } 
            },
            { "$project": { "corpusid": 1 }}
            ])))
        
        if len(venue_df) > 0:
    
            corpusIds_we_want = venue_df.corpusid.tolist()

            # If we want only sections containing an occurence of `data`, first grab al the texts 
            text_subset = []
            # all_texts = {}

            for cid in tqdm(corpusIds_we_want):
                # cid = corpusIds_we_want[0]
                # text_subset.append(db.s2orc.find_one({'corpusid': cid}))
                text = db.s2orc.find_one({'corpusid': cid})

                if text is not None and text['content']['annotations']['paragraph'] is not None:
                    current_text = text['content']['text']
                    section_headers_raw = text['content']['annotations']['sectionheader']
                    par_raw = text['content']['annotations']['paragraph']
                    if section_headers_raw:
                        headers_start_end = [(int(_['start']), int(_['end'])) for _ in json.loads(section_headers_raw)]
                        headers_title = [current_text[start:end] for start, end in headers_start_end]
                        header_lookup = {loc[0]: title for loc, title in zip(headers_start_end, headers_title)}
                        
                    par_start_end = [(int(_['start']), int(_['end'])) for _ in json.loads(par_raw)]
                    
                    tot_para = len(par_start_end)
                    new_text = []
                    par_ids = []
                    for j,p in enumerate(par_start_end):
                        # print(j)
                        start_par, end_par = p[0], p[1]
                        prev_section = 'PREFACE'
                        if section_headers_raw:
                            for start_section, section_name in header_lookup.items():
                                if start_par < start_section:
                                    # print(prev_section)
                                    new_text.append(current_text[start_par:end_par])
                                    par_ids.append(j)
                                    break
                                prev_section = section_name
                        else:
                            new_text.append(current_text[start_par:end_par])
                            par_ids.append(j)

                    docs = list(nlp.pipe(new_text))
                    
                    for j, doc in enumerate(docs):
                        tok_text = [w.text for w in doc]
                        text_subset.append({'corpusid': cid, 'venue': venue, 'par_id': j, 'text': tok_text})
                                               
            if len(text_subset) > 0:
                db.s2orc_dark_data.insert_many( text_subset ) 

        i += 1


if __name__ == '__main__':
    main()
