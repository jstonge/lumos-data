import pandas as pd
from datasets import Dataset
from label_studio import labelStudio


def like_alpaca(x):
    """
    Similar to https://huggingface.co/datasets/tatsu-lab/alpaca#data-instances
    also called data-input format. 
    See https://pytorch.org/torchtune/stable/api_ref_datasets.html#datasets
    """
    return {
        "instruction": "Is data availability statement",
        "input": x["text"],
        "output": x["sentiment"],
        "text": f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n\
                    ### Instruction:\nIs data availability statement\n\n\
                    ### Input:\n{x['text']}\n\n\
                    ### Response:\n{x['sentiment']}""",
    }

def main():
    LS = labelStudio()
    
    current_annots = LS.get_annotations(only_annots=True, only_consensus=True)
    
    df = pd.DataFrame(current_annots)

    # extract text column from data 
    df['text'] = df.data.map(lambda x: x["text"])
    

    # why in some cases choices is not a list??
    df['sentiment'] = df.annotations.map(lambda x: x[0]['result'][0]['value']['choices'])
    # remove "empty annotations" (should be done in class)
    df = df[df.sentiment.map(len) > 0]
    # unpack
    df['sentiment'] = df.sentiment.map(lambda x: x[0] if isinstance(x, list) else x)
    
    # tidy data
    df = df[~df['sentiment'].str.contains('nan')]
    df = df[df['sentiment'] != 'maybe']
    df = df.drop_duplicates(subset='text')

    df['sentiment_encoded'] = df.sentiment.map({'yes': 1, 'no': 0})
    
    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    ds = Dataset.from_pandas(df[['text', 'sentiment_encoded', 'sentiment']], preserve_index=False).train_test_split(test_size=0.2, seed=42)
    
    ds.push_to_hub(f"jstonge1/data-statements-{today}")
    ds.save_to_disk(f"../data/annots/data-statements-{today}")

    # format same data for fine-tuning
    updated_dataset = ds.map(like_alpaca,  remove_columns=ds.column_names['train'])

    updated_dataset.push_to_hub("data-statements-clean-2024-05-31")
    

if __name__ == "__main__":
    main()
    