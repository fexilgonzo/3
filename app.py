import os
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from datasets import load_dataset
import io
import json
from sklearn.model_selection import train_test_split
import datasets
import pandas as pd
from huggingface_hub import login

login(token="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")

# Dataset functions
def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

# Load Mistral model
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True
    )

FastLanguageModel.for_inference(model)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

tokenizer.pad_token = tokenizer.eos_token
EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass


dataset_path = "/work/lpr689/lpr689/alpaca_data.json"
data = jload(dataset_path)

dataset = load_dataset("json", data_files=dataset_path)

dataset = dataset.map(formatting_prompts_func, batched=True)

train_dataset, test_dataset = train_test_split(dataset["train"], test_size=0.2, random_state=42)

train_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=train_dataset))
test_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=test_dataset))

# Before training
def generate_text(text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print("Before training\n")
generate_text("list the top 5 most popular movies of all time")

# 4. Do model patching and add fast LoRA weights and training
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  # Rank stabilized LoRA
    loftq_config = None, # LoftQ
)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 60,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
        optim = "adamw_8bit",
        seed = 3407,
    ),
)
trainer.train()

print("\n ######## \nAfter training\n")
generate_text("List the top 5 most popular movies of all time.")

# 5. Save and push to Hub
model.save_pretrained("lora_model")
model.save_pretrained_merged("outputs", tokenizer, save_method = "merged_16bit",)
