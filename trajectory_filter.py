import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_INPUT_FILE = "output/trajectory_extra.csv"
DEFAULT_OUTPUT_FILE = "output/trajectory_filtered.csv"
DEFAULT_METRICS_FILE = "results/trajectory_filter_metrics.csv"

DT = 1.0
TRAJECTORY_POINTS = 6
SAMPLED_POINTS = (2, 4, 6)
HORIZON_PAIRS = {
    "1s": 1,
    "2s": 2,
    "3s": 3,
}

Point = Tuple[float, float]


def rotate(point: Point, angle: float) -> Point:
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    return (
        cos_angle * point[0] - sin_angle * point[1],
        sin_angle * point[0] + cos_angle * point[1],
    )


def predict_ct_state(state: List[float], dt: float) -> List[float]:
    """Predict [px, py, vx, vy, ax, ay, yaw_rate] with a CT motion model."""
    px, py, vx, vy, ax, ay, yaw_rate = state
    dx = vx * dt + 0.5 * ax * dt * dt
    dy = vy * dt + 0.5 * ay * dt * dt
    next_vx = vx + ax * dt
    next_vy = vy + ay * dt

    if abs(yaw_rate) > 1e-8:
        angle = yaw_rate * dt
        dx, dy = rotate((dx, dy), 0.5 * angle)
        next_vx, next_vy = rotate((next_vx, next_vy), angle)
        ax, ay = rotate((ax, ay), angle)

    return [px + dx, py + dy, next_vx, next_vy, ax, ay, yaw_rate]


def update_position(
    state: List[float],
    covariance: List[float],
    measurement: Point,
    measurement_variance: float,
) -> None:
    """Apply a simple diagonal Kalman position update in place."""
    for state_index, observed_value in ((0, measurement[0]), (1, measurement[1])):
        innovation = observed_value - state[state_index]
        gain = covariance[state_index] / (covariance[state_index] + measurement_variance)
        state[state_index] += gain * innovation
        covariance[state_index] *= 1.0 - gain


def kalman_filter_trajectory(
    initial_velocity: Point,
    initial_acceleration: Point,
    yaw_rate: float,
    measurements: Sequence[Point],
    dt: float = DT,
    process_variance: float = 0.05,
    measurement_variance: float = 0.25,
) -> List[Point]:
    """Filter 1s-sampled trajectory points with a compact CT Kalman filter."""
    state = [
        0.0,
        0.0,
        initial_velocity[0],
        initial_velocity[1],
        initial_acceleration[0],
        initial_acceleration[1],
        yaw_rate,
    ]
    covariance = [0.01, 0.01, 1.0, 1.0, 1.0, 1.0, 0.1]

    filtered = []
    for measurement in measurements:
        state = predict_ct_state(state, dt)
        covariance = [value + process_variance for value in covariance]
        update_position(state, covariance, measurement, measurement_variance)
        filtered.append((state[0], state[1]))

    return filtered


def read_full_trajectory(row: Dict[str, str]) -> List[Point]:
    return [
        (float(row[f"traj_x{index}"]), float(row[f"traj_y{index}"]))
        for index in range(1, TRAJECTORY_POINTS + 1)
    ]


def sample_trajectory(trajectory: Sequence[Point]) -> List[Point]:
    return [trajectory[index - 1] for index in SAMPLED_POINTS]


def calc_l2(traj1: Sequence[Point], traj2: Sequence[Point], first_n_pairs: int) -> float:
    """Same metric style as evaluation.py: average L2 over the first N pairs."""
    total = 0.0
    for index, ((x1, y1), (x2, y2)) in enumerate(zip(traj1, traj2)):
        if index >= first_n_pairs:
            break
        total += math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return total / first_n_pairs


def output_fieldnames(input_fieldnames: Iterable[str]) -> List[str]:
    fieldnames = list(input_fieldnames)
    for index in SAMPLED_POINTS:
        fieldnames.extend([f"filtered_x{index}", f"filtered_y{index}"])
    fieldnames.extend(["l2_1s", "l2_2s", "l2_3s"])
    return fieldnames


def write_metrics(metrics_file: str, metrics: Dict[str, float], rows: int) -> None:
    output_path = Path(metrics_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["rows", "l2_1s", "l2_2s", "l2_3s", "l2_avg"],
        )
        writer.writeheader()
        writer.writerow({"rows": rows, **metrics})


def filter_trajectory_csv(
    input_file: str = DEFAULT_INPUT_FILE,
    output_file: str = DEFAULT_OUTPUT_FILE,
    metrics_file: str = DEFAULT_METRICS_FILE,
    dt: float = DT,
    process_variance: float = 0.05,
    measurement_variance: float = 0.25,
) -> Dict[str, float]:
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    l2_values = {horizon: [] for horizon in HORIZON_PAIRS}
    row_count = 0

    with input_path.open("r", encoding="utf-8", newline="") as source, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} is missing a CSV header.")

        writer = csv.DictWriter(target, fieldnames=output_fieldnames(reader.fieldnames))
        writer.writeheader()

        for row in reader:
            target_traj = sample_trajectory(read_full_trajectory(row))
            filtered_traj = kalman_filter_trajectory(
                initial_velocity=(float(row["vx"]), float(row["vy"])),
                initial_acceleration=(float(row["ax"]), float(row["ay"])),
                yaw_rate=float(row["heading_angular_velocity"]),
                measurements=target_traj,
                dt=dt,
                process_variance=process_variance,
                measurement_variance=measurement_variance,
            )

            output_row = dict(row)
            for source_index, (filtered_x, filtered_y) in zip(SAMPLED_POINTS, filtered_traj):
                output_row[f"filtered_x{source_index}"] = filtered_x
                output_row[f"filtered_y{source_index}"] = filtered_y

            for horizon, first_n_pairs in HORIZON_PAIRS.items():
                l2 = calc_l2(target_traj, filtered_traj, first_n_pairs)
                output_row[f"l2_{horizon}"] = l2
                l2_values[horizon].append(l2)

            writer.writerow(output_row)
            row_count += 1

    if row_count == 0:
        raise ValueError(f"{input_path} does not contain data rows.")

    metrics = {
        f"l2_{horizon}": sum(values) / len(values)
        for horizon, values in l2_values.items()
    }
    metrics["l2_avg"] = (
        metrics["l2_1s"] + metrics["l2_2s"] + metrics["l2_3s"]
    ) / 3.0
    write_metrics(metrics_file, metrics, row_count)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a compact CT Kalman filter and score with evaluation.py L2."
    )
    parser.add_argument("--input_file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output_file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--metrics_file", default=DEFAULT_METRICS_FILE)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--process_variance", type=float, default=0.05)
    parser.add_argument("--measurement_variance", type=float, default=0.25)
    args = parser.parse_args()

    metrics = filter_trajectory_csv(
        input_file=args.input_file,
        output_file=args.output_file,
        metrics_file=args.metrics_file,
        dt=args.dt,
        process_variance=args.process_variance,
        measurement_variance=args.measurement_variance,
    )
    print(f"Saved filtered trajectories to {args.output_file}")
    print(f"Saved L2 metrics to {args.metrics_file}")
    print(
        "L2: "
        f"1s={metrics['l2_1s']:.6f}, "
        f"2s={metrics['l2_2s']:.6f}, "
        f"3s={metrics['l2_3s']:.6f}, "
        f"avg={metrics['l2_avg']:.6f}"
    )


if __name__ == "__main__":
    main()
