import argparse
import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_SPLITS = (
    ("data/train_with_token.json", "data/train.json"),
    ("data/val_with_token.json", "data/val.json"),
)


def remove_token_field(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return one sample with the top-level token field removed."""
    if "messages" not in record:
        raise ValueError("Input record is missing required 'messages' field.")

    return {key: value for key, value in record.items() if key != "token"}


def extract_messages(input_file: str, output_file: str) -> None:
    """Read NDJSON data, remove top-level token fields, and write NDJSON output."""
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as target:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {input_path} at line {line_number}."
                ) from exc

            output_record = remove_token_field(record)
            target.write(json.dumps(output_record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove top-level token fields from train/validation message data."
    )
    parser.add_argument(
        "--train_input",
        default=DEFAULT_SPLITS[0][0],
        help="Training data file with token fields.",
    )
    parser.add_argument(
        "--train_output",
        default=DEFAULT_SPLITS[0][1],
        help="Training data output file without token fields.",
    )
    parser.add_argument(
        "--val_input",
        default=DEFAULT_SPLITS[1][0],
        help="Validation data file with token fields.",
    )
    parser.add_argument(
        "--val_output",
        default=DEFAULT_SPLITS[1][1],
        help="Validation data output file without token fields.",
    )
    args = parser.parse_args()

    extract_messages(args.train_input, args.train_output)
    extract_messages(args.val_input, args.val_output)


if __name__ == "__main__":
    main()
