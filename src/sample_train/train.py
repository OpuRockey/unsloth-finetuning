
import json
from pathlib import Path

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "train.jsonl"
VALIDATION_PATH = PROJECT_ROOT / "data" / "valid.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "lora_adapter"

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 1024

# A short smoke test. Increase this after the pipeline works.
MAX_STEPS = 150


# --------------------------------------------------
# 2. Validate the JSONL dataset
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

with DATA_PATH.open("r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        if not line.strip():
            continue

        example = json.loads(line)

        if not isinstance(example.get("messages"), list):
            raise TypeError(
                f"Line {line_number}: expected a messages array"
            )

        if len(example["messages"]) < 2:
            raise ValueError(
                f"Line {line_number}: expected at least two messages"
            )

        for message in example["messages"]:
            if message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError(
                    f"Line {line_number}: invalid message role"
                )

            if not isinstance(message.get("content"), str):
                raise TypeError(
                    f"Line {line_number}: content must be a string"
                )

print("Loading dataset...")

dataset = load_dataset(
    "json",
    data_files=str(DATA_PATH),
    split="train",
)

validation_dataset = load_dataset(
    "json",
    data_files=str(VALIDATION_PATH),
    split="train",
)

if len(dataset) == 0:
    raise ValueError("The training dataset is empty.")

if len(validation_dataset) == 0:
    raise ValueError("The validation dataset is empty.")

print(f"Training examples: {len(dataset)}")
print(f"Validation examples: {len(validation_dataset)}")

# --------------------------------------------------
# 3. Verify GPU
# --------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Training stopped.")

print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# 4. Load pretrained model and tokenizer
# --------------------------------------------------

print("Loading pretrained model...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

# Use the model's native chat template.
if tokenizer.chat_template is None:
    raise RuntimeError(
        "The tokenizer has no chat template. "
        "Select a compatible instruct model."
    )


# --------------------------------------------------
# 5. Configure LoRA adapters
# --------------------------------------------------

print("Configuring LoRA...")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

model.print_trainable_parameters()


# --------------------------------------------------
# 6. Configure supervised fine-tuning
# --------------------------------------------------

training_args = SFTConfig(
    output_dir=str(PROJECT_ROOT / "outputs" / "checkpoints"),
    max_length=MAX_SEQ_LENGTH,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=MAX_STEPS,
    learning_rate=2e-4,
    warmup_steps=2,
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=10,
    save_strategy="steps",
    save_steps=10,
    save_total_limit=2,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    report_to="none",
    dataloader_num_workers=0,
)

# This version of TRL accepts conversational datasets
# with a messages field and uses the tokenizer's chat template.
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=validation_dataset,
    processing_class=tokenizer,
)


# --------------------------------------------------
# 7. Train
# --------------------------------------------------

print("Starting fine-tuning...")

trainer.train()


# --------------------------------------------------
# 8. Save the adapter
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

print(f"Training completed. Adapter saved to: {OUTPUT_DIR}")
