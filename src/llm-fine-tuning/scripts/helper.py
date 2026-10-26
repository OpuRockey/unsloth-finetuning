from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

ALLOWED_ROLES = {"system", "user", "assistant"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and prepare conversational JSONL data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DATA_DIR / "train.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def fingerprint(messages: list[dict[str, str]]) -> str:
    payload = canonical_json(messages)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_messages(
    record: Any,
) -> tuple[list[dict[str, str]] | None, list[str]]:
    errors: list[str] = []

    if not isinstance(record, dict):
        return None, ["Record must be a JSON object."]

    messages = record.get("messages")

    if not isinstance(messages, list) or not messages:
        return None, ["'messages' must be a non-empty list."]

    normalized: list[dict[str, str]] = []

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"Message {index}: must be an object.")
            continue

        role = message.get("role")
        content = message.get("content")

        if role not in ALLOWED_ROLES:
            errors.append(f"Message {index}: unsupported role {role!r}.")
            continue

        if not isinstance(content, str) or not content.strip():
            errors.append(f"Message {index}: content must be a non-empty string.")
            continue

        normalized.append({"role": role, "content": content.strip()})

    if errors:
        return None, errors

    if not any(m["role"] == "user" for m in normalized):
        errors.append("Conversation must contain a user message.")

    if not any(m["role"] == "assistant" for m in normalized):
        errors.append("Conversation must contain an assistant message.")

    if normalized and normalized[0]["role"] == "assistant":
        errors.append("Conversation must not start with an assistant message.")

    if normalized and normalized[-1]["role"] != "assistant":
        errors.append("Conversation must end with an assistant message.")

    # Check the order of conversational turns.
    previous_role: str | None = None

    for index, message in enumerate(normalized):
        role = message["role"]

        if role == "system":
            if index != 0:
                errors.append(f"Message {index}: system messages must appear first.")
            continue

        if role == previous_role:
            errors.append(
                f"Message {index}: consecutive {role!r} messages are not allowed."
            )

        previous_role = role

    if errors:
        return None, errors

    return normalized, []


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                rejected_records.append(
                    {
                        "line": line_number,
                        "errors": [f"Invalid JSON: {exc.msg}"],
                        "raw": raw_line.rstrip("\r\n"),
                    }
                )
                continue

            messages, errors = validate_messages(record)

            if errors:
                rejected_records.append(
                    {
                        "line": line_number,
                        "errors": errors,
                        "record": record,
                    }
                )
                continue

            assert messages is not None

            valid_records.append(
                {
                    "line": line_number,
                    "messages": messages,
                    "fingerprint": fingerprint(messages),
                }
            )

    return valid_records, rejected_records


def remove_duplicates(
    records: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    duplicate_count = 0

    for record in records:
        key = record["fingerprint"]

        if key in seen:
            duplicate_count += 1
            rejected.append(
                {
                    "line": record["line"],
                    "errors": ["Exact duplicate conversation."],
                    "record": {"messages": record["messages"]},
                }
            )
            continue

        seen.add(key)
        unique.append(record)

    return unique, duplicate_count


def count_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    output = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    token_ids = output["input_ids"] if hasattr(output, "input_ids") else output
    return len(token_ids)


def split_records(
    records: list[dict[str, Any]],
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = records.copy()
    random.Random(seed).shuffle(shuffled)

    if len(shuffled) < 2:
        raise ValueError(
            "At least two usable unique conversations are needed "
            "to create non-empty training and validation sets."
        )

    validation_count = round(len(shuffled) * validation_ratio)
    validation_count = max(1, min(validation_count, len(shuffled) - 1))

    validation = shuffled[:validation_count]
    training = shuffled[validation_count:]

    return training, validation


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(
                json.dumps(
                    {"messages": record["messages"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
