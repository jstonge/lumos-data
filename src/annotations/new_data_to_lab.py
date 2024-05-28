# import jsonlines
import pandas as pd
import re
import json

from pathlib import Path
from creds import client
from tqdm import tqdm


ROOT_DIR = Path("..")
OUTPUT_DIR = ROOT_DIR / "output"
MISC_DIR = ROOT_DIR / "etc"

db = client['papersDB']

list(db.s2orc_dark_data.aggregate([
    { "$match": {"text": "data"}  },
    { "$limit": 1 }
    ])
)

pipeline = [
    { "$match": {"text": "data"}  },
    {
         '$group': {
               '_id': { 'venue': '$venue' }, 
               'count': { '$sum': 1 }
         }
    },
    {
          "$match": {
            "count": {"$gt": 500}
      } 
    },
    {
        "$project" : {
            "_id" : '$_id.venue'
      }
    }
   ]

list_venues = pd.DataFrame(list(db.s2orc_dark_data.aggregate(pipeline)))
all_jns_gt_thresh = set(list_venues['_id'].sort_values().tolist())

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

chosen_journals - all_jns_gt_thresh


out = []
for jn in chosen_journals:
    pipeline2 = [
        { "$match":  { "venue": jn, "text": "data" }   } ,
        { "$sample": { "size": 2000 } }
    ]

    out.append(pd.DataFrame(list(db.s2orc_dark_data.aggregate(pipeline2))))

out = pd.concat(out, axis=0)

out.value_counts("venue")

from textwrap import wrap
wrap(' '.join(out.text.iloc[10]))


  
  
  
  
  
  
  

  
  
  
  
  


  
  
  
  
  
  
  
  

  
  
  
  

  
  
  

  

  
  
  
  

  


  
  


  
  
  
  
  

  
  

  
  
  
  
  
  

  
  
  
  
  
  
  
  

  
  
  
  
  


  

  
  
  
  

  
  
  
  
  
  
  
  

  
  
  
  
  
  
  
  
  
  
  

  
  
  
  
  
  
  
  
  
  
  
  
  



  
  
  
  

  

  
  
  
  
  
  
  
  



  
  
  
  
  
  
  
  
  
  
