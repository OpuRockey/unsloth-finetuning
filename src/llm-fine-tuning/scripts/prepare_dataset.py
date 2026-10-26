from __future__ import annotations

import json
from typing import Any

from helper import (
    count_tokens,
    parse_args,
    read_jsonl,
    remove_duplicates,
    split_records,
    write_jsonl,
)
from transformers import AutoTokenizer


def main() -> None:
    args = parse_args()

    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be greater than zero.")

    if not 0 < args.validation_ratio < 1:
        raise ValueError("--validation-ratio must be between 0 and 1.")

    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=True,
        cache_dir=str(args.cache_dir),
    )

    if not tokenizer.chat_template:
        raise ValueError(
            "The selected tokenizer has no chat template. "
            "Choose a compatible instruct/chat model tokenizer."
        )

    records, rejected = read_jsonl(args.input)
    input_count = len(records) + len(rejected)

    records, duplicate_count = remove_duplicates(records, rejected)

    accepted: list[dict[str, Any]] = []
    over_limit_count = 0

    for record in records:
        try:
            token_count = count_tokens(tokenizer, record["messages"])
        except Exception as exc:
            rejected.append(
                {
                    "line": record["line"],
                    "errors": [
                        f"Chat-template/tokenization error: {type(exc).__name__}: {exc}"
                    ],
                    "record": {"messages": record["messages"]},
                }
            )
            continue

        record["token_count"] = token_count

        if token_count > args.max_tokens:
            over_limit_count += 1
            rejected.append(
                {
                    "line": record["line"],
                    "errors": [
                        f"Token count {token_count} exceeds limit {args.max_tokens}."
                    ],
                    "record": {"messages": record["messages"]},
                }
            )
            continue

        accepted.append(record)

    training, validation = split_records(
        accepted,
        args.validation_ratio,
        args.seed,
    )

    write_jsonl(args.output_dir / "train.jsonl", training)
    write_jsonl(args.output_dir / "valid.jsonl", validation)

    with (args.output_dir / "rejected.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as file:
        for item in rejected:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = {
        "input_file": str(args.input),
        "tokenizer": args.tokenizer,
        "max_tokens": args.max_tokens,
        "validation_ratio": args.validation_ratio,
        "seed": args.seed,
        "input_records_including_invalid": input_count,
        "valid_records_before_deduplication": input_count
        - sum(1 for item in rejected if "record" not in item and "raw" not in item),
        "exact_duplicates_removed": duplicate_count,
        "over_limit_records": over_limit_count,
        "training_records": len(training),
        "validation_records": len(validation),
        "rejected_records_total": len(rejected),
        "token_count_training": {
            "min": min(r["token_count"] for r in training),
            "max": max(r["token_count"] for r in training),
            "mean": round(sum(r["token_count"] for r in training) / len(training), 2),
        },
        "token_count_validation": {
            "min": min(r["token_count"] for r in validation),
            "max": max(r["token_count"] for r in validation),
            "mean": round(
                sum(r["token_count"] for r in validation) / len(validation), 2
            ),
        },
    }

    with (args.output_dir / "report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\nDataset preparation complete.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
