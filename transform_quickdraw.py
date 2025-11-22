import argparse
import gzip
import json
from typing import List, Tuple

import numpy as np


def process_raw_drawing(
    drawing,
    stroke_gap_ms: float = 50.0,
    add_pen_up: bool = True,
    min_points: int = 5,
) -> np.ndarray:
    """
    Convert one raw QuickDraw drawing into a sequence of VLA-style actions.

    Input:
        drawing: list of strokes, each stroke:
            [
                [x0, x1, ...],
                [y0, y1, ...],
                [t0, t1, ...]   # ms since first point in *this stroke*
            ]
        stroke_gap_ms: synthetic pause between strokes (in ms) for global time.
        add_pen_up: whether to insert a pen-up marker after each stroke.
        min_points: minimum number of flattened points required to keep this sample.

    Output:
        actions: (T-1, 4) array with columns:
            [dx_norm, dy_norm, dt_norm, pen]
        or None if too few points.
    """

    xs: List[float] = []
    ys: List[float] = []
    ts: List[float] = []
    pens: List[float] = []

    # Build a global, monotonically increasing time axis
    global_time = 0.0

    for stroke in drawing:
        if len(stroke) < 3:
            # malformed stroke
            continue

        sx, sy, st = stroke
        n = len(sx)
        if n == 0:
            continue

        # Ensure lengths match
        if not (len(sx) == len(sy) == len(st)):
            continue

        # Add points in this stroke
        for x, y, t in zip(sx, sy, st):
            xs.append(float(x))
            ys.append(float(y))
            ts.append(global_time + float(t))
            pens.append(1.0)  # pen-down while drawing

        # Advance global time by stroke duration
        global_time += float(st[-1])

        # Optional pen-up marker after stroke
        if add_pen_up:
            global_time += stroke_gap_ms
            xs.append(float(sx[-1]))
            ys.append(float(sy[-1]))
            ts.append(global_time)
            pens.append(0.0)  # pen-up

        else:
            global_time += stroke_gap_ms

    if len(xs) < min_points:
        return None

    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    ts = np.asarray(ts, dtype=np.float32)
    pens = np.asarray(pens, dtype=np.float32)

    # Normalize x,y to [0,1] per drawing (shape invariant to device)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    # Guard against degenerate drawings (all points same)
    if xmax - xmin < 1e-6 or ymax - ymin < 1e-6:
        return None

    xs_norm = (xs - xmin) / (xmax - xmin + 1e-6)
    ys_norm = (ys - ymin) / (ymax - ymin + 1e-6)

    # Compute deltas
    dx = xs_norm[1:] - xs_norm[:-1]
    dy = ys_norm[1:] - ys_norm[:-1]

    # Compute real dt between points (global, monotonic)
    dt = ts[1:] - ts[:-1]
    dt = np.clip(dt, 1.0, None)  # avoid zero or negative

    total_duration = ts[-1] - ts[0]
    if total_duration <= 0:
        return None

    dt_norm = dt / (total_duration + 1e-6)

    # Align pen state with actions (action i moves from point i-1 -> i)
    pen = pens[1:]

    # Stack into (T-1, 4)
    actions = np.stack([dx, dy, dt_norm, pen], axis=1)
    return actions


def process_ndjson_file(
    input_path: str,
    output_path: str,
    max_samples: int = None,
    verbose_every: int = 10000,
):
    """
    Read a raw QuickDraw NDJSON file and convert each drawing
    into a VLA-style trajectory. Save all trajectories to a .npz file.
    """

    actions_list: List[np.ndarray] = []
    words: List[str] = []
    key_ids: List[str] = []

    # Support plain or gzipped NDJSON
    if input_path.endswith(".gz"):
        opener = lambda p: gzip.open(p, "rt", encoding="utf-8")
    else:
        opener = lambda p: open(p, "r", encoding="utf-8")

    with opener(input_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            drawing = data.get("drawing", None)
            if drawing is None:
                continue

            actions = process_raw_drawing(drawing)
            if actions is None:
                continue

            actions_list.append(actions)
            words.append(data.get("word", ""))
            key_ids.append(str(data.get("key_id", "")))

            if verbose_every and (len(actions_list) % verbose_every == 0):
                print(f"Processed {len(actions_list)} valid drawings so far...")

            if max_samples is not None and len(actions_list) >= max_samples:
                break

    if not actions_list:
        raise RuntimeError("No valid drawings found in file.")

    # Store as object-arrays to keep variable-length trajectories
    actions_arr = np.array(actions_list, dtype=object)
    words_arr = np.array(words, dtype=object)
    key_ids_arr = np.array(key_ids, dtype=object)

    np.savez_compressed(
        output_path,
        actions=actions_arr,
        words=words_arr,
        key_ids=key_ids_arr,
    )

    print(f"Saved {len(actions_list)} trajectories to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw QuickDraw NDJSON to VLA-style trajectories."
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to raw NDJSON file (per category)."
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Path to output .npz file."
    )
    parser.add_argument(
        "--max-samples",
        "-n",
        type=int,
        default=None,
        help="Optional max number of drawings to process.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_ndjson_file(
        input_path=args.input,
        output_path=args.output,
        max_samples=args.max_samples,
    )
