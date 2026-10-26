
from pathlib import Path

from unsloth import FastLanguageModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = PROJECT_ROOT / "outputs" / "lora_adapter"
GGUF_DIR = PROJECT_ROOT / "outputs" / "gguf"

MAX_SEQ_LENGTH = 1024

print("Loading fine-tuned adapter...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(ADAPTER_DIR),
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

print("Merging LoRA and exporting GGUF (Q8_0)...")
model.save_pretrained_gguf(
    str(GGUF_DIR),
    tokenizer,
    quantization_method="q8_0",
)

print(f"GGUF exported to: {GGUF_DIR}")
