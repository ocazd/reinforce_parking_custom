"""
run_pipeline.py
- Train all configured algorithms sequentially
- Run final comparison evaluation automatically
- Record demo videos for each trained algorithm

Usage:
    python run_pipeline.py
    python run_pipeline.py --total_steps 200000 --n_vehicles 10 --n_episodes 100
    python run_pipeline.py --skip_train --skip_record
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch


# Keep this list consistent with the agents/checkpoints you actually use.
ALGORITHMS = ["dqn", "double_dqn", "reinforce", "a2c", "ppo", "sac"]


def resolve_device(requested: str) -> str:
    """Use CUDA when requested and available; fail fast otherwise."""
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but no GPU is available. "
                "Install a CUDA-enabled PyTorch build or pass --device cpu."
            )
        name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {name}")
        return requested
    return requested


def run_cmd(cmd: list[str]):
    print("\n" + "=" * 80)
    print("RUN:", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)


def script_supports_option(script_path: str, option: str) -> bool:
    """
    Avoid breaking evaluate.py if it has not been updated yet.

    record.py in this version supports --n_vehicles, but evaluate.py may or may not.
    This function reads the local script text and only adds optional args if the
    target script appears to define that argparse option.
    """
    try:
        text = Path(script_path).read_text(encoding="utf-8")
    except OSError:
        return False
    return (f'"{option}"' in text) or (f"'{option}'" in text)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=300_000)
    parser.add_argument("--noise_std", type=float, default=0.02)

    # Default changed to 10 so the environment becomes:
    #   10 parked vehicles = 8 normal vehicles + 2 high-value vehicles
    # when ParkingEnv(high_value_vehicle_count=2) is used.
    parser.add_argument("--n_vehicles", type=int, default=10)

    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--save_every", type=int, default=50_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--record_episodes", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="videos")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training and only run compare evaluation.",
    )
    parser.add_argument(
        "--skip_record",
        action="store_true",
        help="Skip video recording after evaluation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.device = resolve_device(args.device)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[PIPELINE] n_vehicles = {args.n_vehicles}")

    ckpt_dir = os.path.join(args.log_dir, "checkpoints")

    if not args.skip_train:
        for algo in ALGORITHMS:
            run_cmd([
                sys.executable,
                "train.py",
                "--algo", algo,
                "--total_steps", str(args.total_steps),
                "--noise_std", str(args.noise_std),
                "--n_vehicles", str(args.n_vehicles),
                "--log_dir", args.log_dir,
                "--save_every", str(args.save_every),
                "--device", args.device,
            ])

    eval_cmd = [
        sys.executable,
        "evaluate.py",
        "--compare",
        "--log_dir", args.log_dir,
        "--n_episodes", str(args.n_episodes),
        "--noise_std", str(args.noise_std),
        "--device", args.device,
    ]

    # If evaluate.py has also been updated to accept --n_vehicles, keep the
    # evaluation environment consistent with training/recording.
    if script_supports_option("evaluate.py", "--n_vehicles"):
        eval_cmd.extend(["--n_vehicles", str(args.n_vehicles)])
    else:
        print(
            "[WARN] evaluate.py does not appear to support --n_vehicles, "
            "so comparison evaluation may use evaluate.py's default vehicle count. "
            "Send evaluate.py too if you want me to patch it."
        )

    run_cmd(eval_cmd)

    if not args.skip_record:
        for algo in ALGORITHMS:
            ckpt_path = os.path.join(ckpt_dir, f"{algo}_final.pt")
            if not os.path.exists(ckpt_path):
                print(f"[SKIP] record {algo}: checkpoint not found at {ckpt_path}")
                continue

            run_cmd([
                sys.executable,
                "record.py",
                "--algo", algo,
                "--ckpt", ckpt_path,
                "--n_episodes", str(args.record_episodes),
                "--output_dir", args.output_dir,
                "--fps", str(args.fps),
                "--noise_std", str(args.noise_std),
                "--n_vehicles", str(args.n_vehicles),
                "--device", args.device,
            ])

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
