import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig, TextStreamer, pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from textwrap import wrap
from inspect import cleandoc

#####################
#                   #
#  Loading data     #
#                   #
#####################

# some training data, not sure how it has been generated
data = pd.read_parquet("data/annotated_data.parquet")
data["wc"] = data.text.str.split(" ").map(len)
data = data[data.wc <= 600]
data = data[data.sentiment != "maybe"]
data = data[~data.sentiment.isna()]
# balance dat
# data_pos = data[data.sentiment == 'yes']
# data_neg = data[data.sentiment == 'no'].sample(len(data_pos), random_state=42)
# data = pd.concat([data_pos, data_neg])
text = data.text.tolist()


#####################
#                   #
#   Setup model     #
#                   #
#####################

# load the model
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map = 'auto'
)

# set the configs
generation_config = GenerationConfig.from_pretrained(model_id)
generation_config.max_new_tokens = 512
generation_config.temperature = 0.0001
generation_config.do_sample = True
 
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
 
llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    generation_config=generation_config,
    num_return_sequences=1,
    eos_token_id=stop_token_ids,
    streamer=streamer,
)

def create_messages(prompt: str, system_prompt: str = None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": cleandoc(system_prompt)})
    messages.append({"role": "user", "content": cleandoc(prompt)})
    return messages

######################
#                    #
# Zero-shot prompts  #
#                    #
######################

SYSTEM_PROMPT1 = cleandoc(
    """
You're an expert data annotator. You can tell when a sentence is a data availability statement or not. \
A data availability statement describes where the data of a research paper is available. \
You simply respond with "yes" and "no".
"""
)

# Show how it is formatted
# print(tokenizer.apply_chat_template(create_messages(text[1], system_prompt=SYSTEM_PROMPT1), tokenize=False))

# adding the mention that we include where the data is coming from.
SYSTEM_PROMPT2 = cleandoc(
    """
You're an expert data annotator. You can tell when a sentence is a data availability statement or not. \
A data availability statement describes where the data of a research paper is available, or where it was retrieve. \
Often data is available in a repository, website, table or supplementary materials. \
You simply respond with "yes" and "no".
"""
)

# mentioning that its task is to find the statement
SYSTEM_PROMPT3 = cleandoc(
"""
You're an expert data annotator. You can tell when text contains a data availability statement. \
A data availability statement describes where the data of a research paper is available or where it came from. \
The statement often mention if the data is available in a repository, website, database, table, or supplementary materials. \
You simply respond with "yes" and "no".
"""
)

data["llama3_prompt1"] = data.text.map(lambda x: llm(create_messages(x, system_prompt=SYSTEM_PROMPT1))[0]['generated_text'])
data["llama3_prompt2"] = data.text.map(lambda x: llm(create_messages(x, system_prompt=SYSTEM_PROMPT2))[0]['generated_text'])
data["llama3_prompt3"] = data.text.map(lambda x: llm(create_messages(x, system_prompt=SYSTEM_PROMPT2))[0]['generated_text'])

#######################
#                     #
# Few shot examples   #
#                     #
#######################

ex1 = """We thank A. Sachraida, C. Gould and P. J. Kelly for providing us with the experimental data and helpful comments, and S. Tomsovic for a critical discussion."""

ex2 = """River discharge data for the Tully River were obtained from the Queensland Bureau of Meteorology (http://www.bom.gov. au). No long-term in situ salinity data are available from King Reef; therefore, data from the Carton-Giese Simple Ocean Data Assimilation (SODA) reanalysis project were chosen as a longterm monthly resolution SSS dataset. This consists of a combination of observed and modeled data (Carton et al., 2000). Data were obtained from the box centered on 17.5°S (17.25°-17.75°) and 146°E (145.75°-146.25°). SODA version 1.4.2 extends from 1958 to 2001 and uses surface wind products from the European Center for Medium-Range Weather Forecasts 40-year reanalysis (ECMWF ERA 40), which may contain inaccuracies in tropical regions (Cahyarini et al., 2008). The most recent version of SODA (1.4.3) now uses wind data from the Quick-Scat scatterometer, thus providing more accurate data for the tropics (Cahyarini et al., 2008;Carton and Giese, 2008"""

ex3 = """The current results should be considered relative to a few study limitations. The CFS data did not specify the nature of proactive activities that patrol, DRT officers, or investigators were engaged in. Furthermore, although the coding of the ten call categories analyzed were informed by prior research (Wu & Lum, 2017), idiosyncrasies associated with the study departments' method of cataloging and recording call information did not always allow for direct comparisons to prior research on COVID-19's impact on police services. Similarly, measuring proactivity solely through self-initiated activities from CFS data is not a flawless indicator. Officers may engage in proactive work that is not captured in these **data** (Lum, Koper, et al., 2020). However, this method has been established as a reasonable way to distinguish proactivity from reactivity (Lum, Koper, et al., 2020;Wu & Lum, 2017;Zhang and Zhao, 2021)."""

data["llama3_prompt4"] = data.text.map(lambda x: llm(create_messages(f"""
Text: {ex1}
is_data_availability_statement: yes

Text: {ex2}
is_data_availability_statement: yes

Text: {ex3}
is_data_availability_statement: no

Text: {x}
is_data_availability_statement: ?

Give a one word response.
"""
)))

data["llama3_prompt4"] = data["llama3_prompt4"].map(lambda x: x[0]['generated_text'])
data["llama3_prompt4"] = data["llama3_prompt4"].str.lower()

##################
#                #
#    evaluate    #
#                #
##################

def plot_cm(i):
    data_yes_no = data[data.sentiment.isin(['yes', 'no'])]
    y_true = data_yes_no.sentiment.tolist()
    y_pred = data_yes_no['llama3_prompt'+str(i)].tolist()
    cm = confusion_matrix(y_true, y_pred, labels=["yes", "no"])
    tn, fp, fn, tp = cm.ravel()
    print(f"Precision: {tp/(tp+fp)}")
    print(f"Recall: {tp/(tp+fn)}")
    P = (tp+fn) # positives are all true positives + false negative
    N = (tn+fp) # positives are all true positives + false negative
    print(f"Accuracy: {(tp+tn)/(P+N)}")
    tpr = tp/P # normalized true positive rate
    tnr = tn/N
    print(f"Balanced accuracy: {(tpr+tnr)/2}")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["yes", "no"])
    disp.plot()

plot_cm(1)
plot_cm(2)
plot_cm(3)
plot_cm(4)

# Check where the model is failing

def print_random_failure(i):
    df_sample = data[data.sentiment != data["llama3_prompt"+str(i)]].sample(1)
    print(df_sample.corpusid.iloc[0])
    print("\n".join(wrap(df_sample.iloc[0].text)))

print_random_failure(4)