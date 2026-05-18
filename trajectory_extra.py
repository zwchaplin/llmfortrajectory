import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_INPUT_FILE = "data/val.jsonl"
DEFAULT_OUTPUT_FILE = "output/trajectory_extra.csv"
TRAJECTORY_POINTS = 6
TRAJECTORY_DIMS = 2

NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _extract_pair(text: str, pattern: str, field_name: str) -> Tuple[float, float]:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Missing {field_name}.")
    return float(match.group(1)), float(match.group(2))


def _extract_scalar(text: str, pattern: str, field_name: str) -> float:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Missing {field_name}.")
    return float(match.group(1))


def _extract_mission_goal(text: str) -> str:
    match = re.search(r"Mission Goal:\s*([^\r\n]+)", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError("Missing Mission Goal.")
    return match.group(1).strip()


def extract_ego_states(user_prompt: str) -> Dict[str, Any]:
    """Extract ego-state fields and mission goal from a user prompt."""
    velocity = _extract_pair(
        user_prompt,
        rf"Velocity\s*\(vx\s*,\s*vy\):\s*\(\s*({NUMBER_PATTERN})\s*,\s*({NUMBER_PATTERN})\s*\)",
        "Velocity",
    )
    heading_angular_velocity = _extract_scalar(
        user_prompt,
        rf"Heading Angular Velocity\s*\(v_yaw\):\s*\(\s*({NUMBER_PATTERN})\s*\)",
        "Heading Angular Velocity",
    )
    acceleration = _extract_pair(
        user_prompt,
        rf"Acceleration\s*\(ax\s*,\s*ay\):\s*\(\s*({NUMBER_PATTERN})\s*,\s*({NUMBER_PATTERN})\s*\)",
        "Acceleration",
    )

    return {
        "velocity": velocity,
        "heading_angular_velocity": heading_angular_velocity,
        "acceleration": acceleration,
        "mission_goal": _extract_mission_goal(user_prompt),
    }


def _find_list_after_label(text: str, label: str) -> str:
    label_match = re.search(rf"\b{label}\s*:", text, flags=re.IGNORECASE)
    if not label_match:
        raise ValueError(f"Missing {label}.")

    list_start = text.find("[", label_match.end())
    if list_start == -1:
        raise ValueError(f"Missing list value after {label}.")

    depth = 0
    for index in range(list_start, len(text)):
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[list_start : index + 1]

    raise ValueError(f"Unclosed list value after {label}.")


def extract_trajectory(assistant_prompt: str) -> List[Tuple[float, float]]:
    """Extract Trajectory or nTrajectory data from an assistant prompt."""
    try:
        trajectory_text = _find_list_after_label(assistant_prompt, "nTrajectory")
    except ValueError:
        trajectory_text = _find_list_after_label(assistant_prompt, "Trajectory")

    trajectory = ast.literal_eval(trajectory_text)
    if len(trajectory) != TRAJECTORY_POINTS:
        raise ValueError(
            "Trajectory must have shape "
            f"({TRAJECTORY_POINTS}, {TRAJECTORY_DIMS}), got {len(trajectory)} points."
        )

    trajectory_pairs = []
    for point in trajectory:
        if len(point) != TRAJECTORY_DIMS:
            raise ValueError(
                "Each trajectory point must have "
                f"{TRAJECTORY_DIMS} values, got {len(point)}."
            )
        trajectory_pairs.append((float(point[0]), float(point[1])))

    return trajectory_pairs


def _message_content(messages: Iterable[Dict[str, Any]], role: str) -> str:
    for message in messages:
        if message.get("role") == role:
            return message.get("content", "")
    raise ValueError(f"Missing {role} message.")


def extract_trajectory_rows(input_file: str = DEFAULT_INPUT_FILE) -> List[Dict[str, Any]]:
    """Read JSONL messages and return extracted rows."""
    input_path = Path(input_file)
    rows = []

    with input_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                messages = record["messages"]
                user_prompt = _message_content(messages, "user")
                assistant_prompt = _message_content(messages, "assistant")

                row = extract_ego_states(user_prompt)
                row["trajectory"] = extract_trajectory(assistant_prompt)
                rows.append(row)
            except (KeyError, TypeError, ValueError, SyntaxError) as exc:
                raise ValueError(
                    f"Failed to extract trajectory data from {input_path} "
                    f"at line {line_number}."
                ) from exc

    return rows


def _csv_fieldnames() -> List[str]:
    fieldnames = [
        "vx",
        "vy",
        "heading_angular_velocity",
        "ax",
        "ay",
        "mission_goal",
    ]
    for index in range(1, TRAJECTORY_POINTS + 1):
        fieldnames.extend([f"traj_x{index}", f"traj_y{index}"])
    return fieldnames


def _flatten_row(row: Dict[str, Any]) -> Dict[str, Any]:
    velocity = row["velocity"]
    acceleration = row["acceleration"]
    flat_row = {
        "vx": velocity[0],
        "vy": velocity[1],
        "heading_angular_velocity": row["heading_angular_velocity"],
        "ax": acceleration[0],
        "ay": acceleration[1],
        "mission_goal": row["mission_goal"],
    }

    for index, (traj_x, traj_y) in enumerate(row["trajectory"], start=1):
        flat_row[f"traj_x{index}"] = traj_x
        flat_row[f"traj_y{index}"] = traj_y

    return flat_row


def save_trajectory_csv(input_file: str, output_file: str) -> List[Dict[str, Any]]:
    """Extract rows and save them as a CSV table."""
    rows = extract_trajectory_rows(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = _csv_fieldnames()
    with output_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_flatten_row(row))

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ego-state fields and assistant trajectory data from JSONL "
            "messages into a CSV table."
        )
    )
    parser.add_argument(
        "--input_file",
        default=DEFAULT_INPUT_FILE,
        help="Input JSONL file containing messages.",
    )
    parser.add_argument(
        "--output_file",
        default=DEFAULT_OUTPUT_FILE,
        help="Output CSV file.",
    )
    args = parser.parse_args()

    rows = save_trajectory_csv(args.input_file, args.output_file)
    print(f"Saved {len(rows)} rows to {args.output_file}")


if __name__ == "__main__":
    main()
