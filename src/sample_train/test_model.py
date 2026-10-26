
from pathlib import Path

from unsloth import FastLanguageModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_PATH = PROJECT_ROOT / "outputs" / "lora_adapter"

if not (ADAPTER_PATH / "adapter_config.json").exists():
    raise FileNotFoundError(
        "Adapter not found. Run src/train.py first."
    )

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(ADAPTER_PATH),
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

messages = [
    {
        "role": "user",
        "content": "Who won the football World Cup last year?"
    }
]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

outputs = model.generate(
    input_ids=input_ids,
    max_new_tokens=150,
    do_sample=False,
)

# Decode only the newly generated tokens.
new_tokens = outputs[0][input_ids.shape[-1]:]
answer = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True,
)

print("\nModel response:\n")
print(answer)
