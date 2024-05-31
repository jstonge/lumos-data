from pymongo import MongoClient
import os
import requests
import json
from typing import List, Any, Union, Dict, ClassVar, Set
import pandas as pd
import numpy as np
from pathlib import Path

chosen_journals = set(['Journal of Business Research','Technological Forecasting and Social Change',
                      'Journal of Business Ethics','Advanced Materials', 'Angewandte Chemie', 
                      'Advanced Energy Materials', 'Neural Information Processing Systems', 
                      'International Conference on Learning Representations', 'Journal of Cleaner Production', 
                      'International Journal of Molecular Sciences', 'Nature Medicine', 
                      'BMJ', 'Synthese', 'Digital journalism', 
                      'Media, Culture & Society',  'Science of The Total Environment',  
                      'Nucleic Acids Research',  'International Journal of Molecular Sciences', 
                      'The astrophysical journal',  'Light-Science & Applications',  'Journal of High Energy Physics', 
                      'Nature Human Behaviour', 'Social Science & Medicine', 'Cities'])

class labelStudio:
    
    def __init__(self, client: MongoClient = None):
        if client is None:
            uri="mongodb://cwward:password@wranglerdb01a.uvm.edu:27017/?authSource=admin&readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false"
            client = MongoClient(uri)
        
        self.LS_TOK = os.getenv('LS_TOK')
        self.db = client['papersDB']
        self.cache = Path("./cache")
        self.annotators = {'juniper.lovato@uvm.edu': 19456, 'achawla1@uvm.edu': 17284, 'CW': 23575, 'JZ': 23576, 'jonathan.st-onge@uvm.edu': 16904}
        self.active_annotators = {'juniper.lovato@uvm.edu':19456, 'achawla1@uvm.edu': 17284, 'jonathan.st-onge@uvm.edu': 16904}

        if self.cache.exists() == False:
            self.cache.mkdir()

        print('accessing project status...')
        project_status = self.is_dark_data_project_exists()
        if project_status is None:
            self.proj_id = self.create_dark_data_project()
        else:
            self.proj_id = project_status

    # LABEL STUDIO HELPERS

    def get_annotations(self, only_annots=True):
        """Get annotations of a given project id."""
        headers = { "Authorization": f"Token {self.LS_TOK}" }
        base_url = "https://app.heartex.com/api/projects"
        
        url = f"{base_url}/{self.proj_id}/export?exportType=JSON&download_all_tasks=true"
        print("requesting the annots...")
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            json_data = json.loads(response.text)
            
            if only_annots:
                return [_ for _ in json_data if len(_['annotations']) > 0]
            else:
                return json_data
            
        else:
            print(f"Failed to fetch data: {response.status_code}")

    def post_LS(self, data: List) -> None:
        """Import data to label studio"""
        response = requests.post(f'https://app.heartex.com/api/projects/{self.proj_id}/import', 
                                headers={'Content-Type': 'application/json', 'Authorization': f"Token {self.LS_TOK}"}, 
                                data=json.dumps(data), verify=False)
        print(response.status_code)

    # PROJECT HELPERS

    def is_dark_data_project_exists(self) -> int:
        """if it exists, return project id, else return None"""
        headers = { "Authorization": f"Token {self.LS_TOK}"}
        response = requests.get("https://app.heartex.com/api/projects", headers=headers)
        if response.status_code == 200:
            all_projects = json.loads(response.text)['results']
            proj_id = [_['id'] for _ in all_projects if _['title'] == 'Dark-Data']
            if len(proj_id) > 0:
                return proj_id[0]

    # MONGODB HELPERS

    def read_cache_venue(self, N=500) -> pd.DataFrame:
        """read cache of venues with a certain threshold"""
        cache_f = self.cache / f"venues_{N}.csv"
        if cache_f.exists():
            return pd.read_csv(cache_f)

    def get_venues_more_than_N(self, keyword: str = "data", N:int = 500) -> List[str]:
        """return venues with more than N occurences of a given keyword in text."""
        # the pipeline below takes time. We use caching when we can. 
        list_venues = self.read_cache_venue(N=N)
        
        if list_venues is None:
            print("no cached filed found, querying mongoDB...")
            pipeline = [
                { "$match": {"text": keyword}  },
                {
                    '$group': {
                        '_id': { 'venue': '$venue' }, 
                        'count': { '$sum': 1 }
                    }
                },
                {
                    "$match": {
                        "count": {"$gt": N}
                } 
                },
                {
                    "$project" : {
                        "_id" : '$_id.venue'
                }
                }
            ]
            
            list_venues = pd.DataFrame(list(self.db.s2orc_dark_data.aggregate(pipeline)))
        
        return list(set(list_venues['_id'].sort_values().tolist()))
        
    def get_sample_jn(self, jn:str, size:int, done_ids:Set = set()) -> pd.DataFrame:
        pipeline = [
                { "$match":  { "venue": jn, "text": "data" }   } ,
                { "$sample": { "size": size } }
        ]
        print("querying DB for more pubs...")
        hits = pd.DataFrame(list(self.db.s2orc_dark_data.aggregate(pipeline)))
        
        if len(hits) > 0:
            return hits[~hits['corpusid'].isin(done_ids)]
        else:
            print("no more hits")

        
    # CREATE PROJECT

    def create_dark_data_project(self) -> int:
        """create dark data project"""
        if self.is_dark_data_project_exists():
            return "project already exists"
        
        project_config = """\
            {"title": "Dark-Data","label_config": "<View>\
            <View>\
            <Text name='text' value='$text'/>\
            <Choices name='sentiment' toName='text'>\
                <Choice value='yes'/>\
                <Choice value='no'/>\
                <Choice value='maybe'/>\
            </Choices>\
            </View>\
            </View>"}
            """
    
        response = requests.post(f'https://app.heartex.com/api/projects', 
                         headers={'Content-Type': 'application/json', 'Authorization': f"Token {self.LS_TOK}"}, 
                         data=project_config, verify=False)
        
        if response.text == 200:
            print("project created")
            return json.loads(response.text)['id']

    # WORKING WITH LABEL STUDIO
    
    def more_annotations(self, min_wc:int = 5, max_wc:int = 1000, sample_by_field:int = 1000) -> pd.DataFrame:
        print("get already doing annotations...")
        done_annots = self.get_annotations()
        done_corpusids = set([_['data']['corpusid'] for _ in done_annots]) if len(done_annots) > 0 else set()
        
        print("getting venues with more than 2000 occurences of 'data'...")
        all_jns_gt_thresh = self.get_venues_more_than_N(N=2000)
        
        # We get more annotations for each venue
        out = []
        overshoot_sample = sample_by_field*2 # there must be a better way to do this
        for jn in all_jns_gt_thresh:
            out.append(self.get_sample_jn(jn=jn, size=overshoot_sample, done_ids=done_corpusids))
        new_annots = pd.concat(out, axis=0)

        # just making sure we don't have any overlap
        assert len(set(new_annots.corpusid.unique()) & done_corpusids) == 0
        
        
        # We filter by word count

        new_annots['text'] = new_annots['text'].map(lambda x: ' '.join(x))
        new_annots['wc'] = new_annots.text.str.split(" ").map(len)

        subset_d = new_annots.sample(frac=1, random_state=42)\
                    .groupby('venue').head(sample_by_field)\
                    .query(f'wc > {min_wc} & wc < {max_wc}').reset_index(drop=True)
        
        subset_d['corpusid_unique'] = subset_d.corpusid.astype(str) + '_' + subset_d.par_id.astype(str)
        
        return subset_d
    
    def dispatch_annots(self):
        annots_to_dispatch = self.get_annotations(min_wc=5, max_wc=1000, sample_by_field=500)

        for email, annot_id in self.active_annotators.items():
            print(f"dispatching to {email}")
            next_corpus_id = np.random.choice(annots_to_dispatch.corpusid_unique, 3000)
            next_sample_df = annots_to_dispatch[annots_to_dispatch.corpusid_unique.isin(next_corpus_id)]

            data2dump = []
            for i, row in next_sample_df.iterrows():
                
                data2dump.append({
                    "data": {
                        'corpusid': row['corpusid'],
                        'corpusid_unique': row['corpusid_unique'],
                        'par_id': row['par_id'],
                        'wc': row['wc'],
                        'text': row['text']
                    },
                    "annotations": [{
                        'completed_by' : {
                            "id": annot_id,
                            "first_name": "",
                            "last_name": "",
                            "avatar": None,
                            "email": email,
                            "initials": email[:2]
                        },
                        'result': [{
                            'value': {'choices': []},
                            'id': "",
                            "from_name": 'sentiment',
                            'to_name': 'text',
                            'type': 'choices',
                            'origin': 'manual',
                        }]
                        }]
                })     
            
            self.post_LS(data2dump)

        annots_to_dispatch = annots_to_dispatch[~annots_to_dispatch.corpusid_unique.isin(next_corpus_id)]



# LS = labelStudio()
# new_annots = LS.more_annotations()
# new_annots.value_counts("venue")
    