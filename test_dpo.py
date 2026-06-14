import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig

model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("csv", data_files="llm-fine-tuning/pharma_preference_data.csv", split="train")

def process_data(example):
    return {
        "prompt": f"### Instruction:\n{example['prompt']}\n### Response:\n",
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }

dataset = dataset.map(process_data)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training Arguments using DPOConfig without max_prompt_length
training_args = DPOConfig(
    output_dir="./test-dpo-output",
    num_train_epochs=1,
    max_steps=1,
    per_device_train_batch_size=1,
    learning_rate=5e-5,
    bf16=True,
    logging_steps=1,
    report_to="none",
    beta=0.1,
    max_length=256,
    remove_unused_columns=False
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer, # Pass tokenizer here
    peft_config=lora_config
)

print("Starting DPO training test with processing_class...")
trainer.train()
print("DPO training test passed successfully!")
