
from pathlib import Path

from unsloth import FastLanguageModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = PROJECT_ROOT / "outputs" / "lora_adapter"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(ADAPTER_DIR),
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

for question in [
    "Who manufactures the Commander III?",
    "What is the payload capacity of the Commander III robotic arm?",
    "How far can the Commander III arm reach?",
]:
    messages = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    output = model.generate(
        input_ids=inputs,
        max_new_tokens=48,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    print(f"Q: {question}")
    print(f"A: {tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)}")
    print()
