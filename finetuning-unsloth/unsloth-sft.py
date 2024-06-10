import torch
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from textwrap import wrap

max_seq_length = 2048 # Supports automatic RoPE Scaling, so choose any number

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    max_seq_length=max_seq_length,
    dtype=None,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit=False,  # Use 4bit quantization to reduce memory usage. Can be False
)

# datas stuff
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Inputs:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples['text']
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

import pandas as pd
dataset = load_dataset("jstonge1/data-statements-clean-2024-05-31", split = "train")

balanced_df = pd.read_csv("lumos-data/train.csv")
balanced_df = balanced_df.rename(columns = {"Text": "text", "Label": "output"})
dataset = Dataset.from_pandas(balanced_df)

dataset = dataset.map(formatting_prompts_func, batched = True,)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


trainer = SFTTrainer(
    model=model,
    tokenizer = tokenizer,
    dataset_text_field = "text",
    train_dataset=dataset,
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "./test_unsloth",
    )
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# Inference

test_df = load_dataset("jstonge1/data-statements-clean-2024-05-31", split = "test").to_pandas()
test_df_pos = test_df[test_df.output == 'yes']

# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
i=2
inputs = tokenizer(
[
    alpaca_prompt.format(
        test_df_pos.iloc[i].instruction, # instruction
        test_df_pos.iloc[i].text, # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")


outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
reply = tokenizer.batch_decode(outputs)
print('\n'.join(wrap(test_df_pos.iloc[i].text)))
reply[0].split("Response:\n")[-1].replace("<|eot_id|>", "")



model.save_pretrained("dark-data-lora-test") # Local saving
tokenizer.save_pretrained("dark-data-lora-test")

model.push_to_hub("jstonge1/dark-data-lora-balanced") # Online saving
tokenizer.push_to_hub("jstonge1/dark-data-lora-balanced") # Online saving

############################
#                          #
# loading back the model   #
#                          #
############################

if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "jstonge1/dark-data-lora-balanced", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = False,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

test_df = load_dataset("jstonge1/data-statements-clean-2024-05-31", split = "test").to_pandas()
test_df_pos = test_df[test_df.output == 'yes']

FastLanguageModel.for_inference(model) # Enable native 2x faster inference

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Inputs:
{}

### Response:
{}"""

i=2
inputs = tokenizer(
[
    alpaca_prompt.format(
        test_df_pos.iloc[i].instruction, # instruction
        test_df_pos.iloc[i].text, # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
reply = tokenizer.batch_decode(outputs)
print('\n'.join(wrap(test_df_pos.iloc[i].text)))
reply[0].split("Response:\n")[-1].replace("<|eot_id|>", "")
