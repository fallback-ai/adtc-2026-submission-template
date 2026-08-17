"""Fine-tune Qwen/Qwen2.5-1.5B-Instruct on the English Homa corpus.

English-only pivot: trains on combined_train.en.jsonl (~1,900 rows) using Qwen's
native ChatML format with a fixed Homa system message (Qwen has a real system
role, so identity lives there cleanly — no Gemma-style echo issue).

Run in a GPU env. Requires: transformers, trl, peft, datasets, accelerate.
Output: ./homa-qwen15b-merged  (ready for GGUF conversion).
"""
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
DATA = "../sft_train_samples/combined_train.en.jsonl"
OUT = "homa-qwen15b"
MERGED = "homa-qwen15b-merged"
FULL_FT = False                       # 1.5B full-FT is feasible; try if LoRA underfits
MAX_LEN = 1024
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"

HOMA_SYSTEM = (
    "You are Homa, an offline agricultural assistant for farmers in Nigeria, "
    "built by Fallback AI. You give practical, direct advice on crops, livestock, "
    "soil, pests, weather, and markets."
)

tok = AutoTokenizer.from_pretrained(BASE)

def to_text(row):
    # Qwen ChatML with a fixed system message; identity is reinforced every example.
    return {"text": (f"<|im_start|>system\n{HOMA_SYSTEM}<|im_end|>\n"
                     f"<|im_start|>user\n{row['instruction'].strip()}<|im_end|>\n"
                     f"<|im_start|>assistant\n{row['response'].strip()}<|im_end|>\n")}

ds = load_dataset("json", data_files=DATA, split="train").map(to_text)
ds = ds.remove_columns([c for c in ds.column_names if c != "text"])

model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)

# train only on the assistant turn (mask system + user)
collator = DataCollatorForCompletionOnlyLM(RESPONSE_TEMPLATE, tokenizer=tok)

peft_cfg = None if FULL_FT else LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"])

args = SFTConfig(
    output_dir=OUT,
    num_train_epochs=3,               # small English set + already-instruct base; 3 is safe
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-4 if not FULL_FT else 8e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=20,
    save_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    max_seq_length=MAX_LEN,
    packing=False,
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                     data_collator=collator, peft_config=peft_cfg)
trainer.train()

if FULL_FT:
    trainer.save_model(MERGED)
else:
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(MERGED)
tok.save_pretrained(MERGED)
print(f"done -> ./{MERGED}")
