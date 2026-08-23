"""Fine-tune google/gemma-3-1b-it on the Homa corpus (bake-off vs the 4B).

Reuses combined_train.integrated.jsonl (same 6,765-row corpus as the 4B) and the
same Gemma turn format. LoRA by default; flip FULL_FT=True to full fine-tune (a
1B fits on a single 16-24 GB GPU and full-FT often helps a small model absorb a
big domain + multilingual shift).

Run in a GPU env (Colab/Kaggle). Requires: transformers, trl, peft, datasets,
accelerate, bitsandbytes (optional). Accept the Gemma license on HF first and set
HF_TOKEN. Output: ./homa-gemma1b-merged  (ready for GGUF conversion).
"""
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

BASE = "google/gemma-3-1b-it"           # instruction-tuned base (best for a 1B)
DATA = "../sft_train_samples/combined_train.integrated.jsonl"
OUT = "homa-gemma1b"
MERGED = "homa-gemma1b-merged"
FULL_FT = False                          # True = full fine-tune, False = LoRA
MAX_LEN = 1024
RESPONSE_TEMPLATE = "<start_of_turn>model\n"

tok = AutoTokenizer.from_pretrained(BASE)

def to_text(row):
    # exact Gemma turn format used at inference (matches the Modelfile template)
    return {"text": (f"<start_of_turn>user\n{row['instruction'].strip()}<end_of_turn>\n"
                     f"<start_of_turn>model\n{row['response'].strip()}<end_of_turn>\n")}

ds = load_dataset("json", data_files=DATA, split="train").map(to_text)
ds = ds.remove_columns([c for c in ds.column_names if c != "text"])

model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, attn_implementation="eager")  # eager = stable for gemma training

# train only on the model's turn (mask the prompt)
collator = DataCollatorForCompletionOnlyLM(RESPONSE_TEMPLATE, tokenizer=tok)

peft_cfg = None if FULL_FT else LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"])

args = SFTConfig(
    output_dir=OUT,
    num_train_epochs=2,                  # 1B benefits from a little more; watch overfit
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,       # effective batch ~32
    learning_rate=2e-4 if not FULL_FT else 1e-5,
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

trainer = SFTTrainer(
    model=model, args=args, train_dataset=ds,
    data_collator=collator, peft_config=peft_cfg)
trainer.train()

# merge (LoRA) or save (full), plus tokenizer, into a single HF folder for GGUF
if FULL_FT:
    trainer.save_model(MERGED)
else:
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(MERGED)
tok.save_pretrained(MERGED)
print(f"done -> ./{MERGED}")
