"""
record.py
- Load a trained checkpoint and record episodes as mp4 or gif

Usage:
    python record.py --algo ppo --ckpt logs/checkpoints/ppo_final.pt
    python record.py --algo dqn --ckpt logs/checkpoints/dqn_final.pt --n_episodes 3 --n_vehicles 10
"""

import os
import argparse
import sys
import imageio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import env.parking_env as parking_env_module
from env.parking_env import ParkingEnv
from agents.dqn import DQNAgent
from agents.double_dqn import DoubleDQNAgent
from agents.reinforce import REINFORCEAgent
from agents.a2c import A2CAgent
from agents.ppo import PPOAgent
from agents.sac import SACAgent
from agents.checkpoint_utils import CheckpointMismatchError


def get_success_from_info(info: dict) -> bool:
    """
    Use success computed inside ParkingEnv with true observation.
    Do not recompute success from returned obs because returned obs may contain noise.
    """
    return bool(info.get("is_success", info.get("success", False)))


def save_frames(out_path: str, frames, fps: int) -> str:
    """
    Save frames as mp4 when ffmpeg is available.
    Falls back to gif when mp4 writer is unavailable.
    Returns the actual saved path.
    """
    try:
        imageio.mimsave(out_path, frames, fps=fps, format="FFMPEG")
        return out_path
    except Exception as e:
        gif_path = os.path.splitext(out_path)[0] + ".gif"
        imageio.mimsave(gif_path, frames, duration=1.0 / max(fps, 1))
        print(f"  [WARN] mp4 save failed ({e}). Saved gif instead: {gif_path}")
        return gif_path


def load_agent(algo, ckpt_path, obs_dim, env, device):
    if algo == "dqn":
        agent = DQNAgent(obs_dim=obs_dim, n_actions=env.N_DISCRETE_ACTIONS, device=device)
    elif algo == "double_dqn":
        agent = DoubleDQNAgent(obs_dim=obs_dim, n_actions=env.N_DISCRETE_ACTIONS, device=device)
    elif algo == "reinforce":
        agent = REINFORCEAgent(obs_dim=obs_dim, action_dim=env.action_space.shape[0], device=device)
    elif algo == "a2c":
        agent = A2CAgent(obs_dim=obs_dim, action_dim=env.action_space.shape[0], device=device)
    elif algo == "ppo":
        agent = PPOAgent(obs_dim=obs_dim, action_dim=env.action_space.shape[0], device=device)
    elif algo == "sac":
        agent = SACAgent(obs_dim=obs_dim, action_dim=env.action_space.shape[0], device=device)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    agent.load(ckpt_path)
    return agent


def print_env_debug(env: ParkingEnv, requested_n_vehicles: int):
    """Print actual environment counts after reset for debugging."""
    base = getattr(env, "_base_env", None)

    print(f"[ENV] using file: {parking_env_module.__file__}")
    print(f"[ENV] requested n_vehicles: {requested_n_vehicles}")

    if base is None:
        print("[ENV] base env not found")
        return

    print(f"[ENV] config vehicles_count: {base.config.get('vehicles_count', None)}")

    parked_meta = getattr(base, "parked_vehicle_metadata", [])
    occupied_spots = getattr(base, "occupied_spots", [])
    valid_goal_slots = getattr(base, "valid_goal_slot_infos", None)
    empty_slot_infos = getattr(base, "empty_slot_infos", None)

    high_value_count = sum(1 for m in parked_meta if m.get("is_high_value", False))
    normal_count = len(parked_meta) - high_value_count

    print(f"[ENV] occupied_spots: {len(occupied_spots)}")
    print(f"[ENV] parked vehicles: {len(parked_meta)}")
    print(f"[ENV] normal vehicles: {normal_count}")
    print(f"[ENV] high-value vehicles: {high_value_count}")

    if valid_goal_slots is not None:
        print(f"[ENV] valid goal slots: {len(valid_goal_slots)}")
    elif empty_slot_infos is not None:
        print(f"[ENV] empty slot infos: {len(empty_slot_infos)}")


def record_episodes(algo, ckpt_path, n_episodes, output_dir, device, noise_std, n_vehicles, fps, debug_env):
    os.makedirs(output_dir, exist_ok=True)

    discrete = (algo in ["dqn", "double_dqn"])
    env = ParkingEnv(
        discrete=discrete,
        noise_std=noise_std,
        n_other_vehicles=n_vehicles,
        render_mode="rgb_array",
    )
    obs_dim = env.observation_space.shape[0]

    try:
        agent = load_agent(algo, ckpt_path, obs_dim, env, device)
    except CheckpointMismatchError as e:
        print(f"[ERROR] {e}")
        env.close()
        sys.exit(1)

    print(f"Loaded {algo.upper()} from {ckpt_path}")

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)

        if ep == 0 and debug_env:
            print_env_debug(env, n_vehicles)

        frames = []
        done = False
        ep_reward = 0.0
        info = {}

        while not done:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            if algo in ["dqn", "double_dqn"]:
                action = agent.select_action(obs, training=False)
            else:
                action, _, _ = agent.select_action(obs, training=False)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        success = get_success_from_info(info)
        status = "SUCCESS" if success else "FAIL"

        out_path = os.path.join(output_dir, f"{algo}_ep{ep + 1}_{status}.mp4")
        if frames:
            saved_path = save_frames(out_path, frames, fps)
            print(f"  Episode {ep + 1}: reward={ep_reward:.1f}  {status}  -> {saved_path}")
        else:
            print(f"  Episode {ep + 1}: no frames captured")

    env.close()
    print(f"\nDone. Videos saved to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True,
                        choices=["dqn", "double_dqn", "reinforce", "a2c", "ppo", "sac"])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="videos")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--noise_std", type=float, default=0.0)

    # Default changed to 10 so direct calls to record.py match the intended lot:
    #   10 parked vehicles = 8 normal vehicles + 2 high-value vehicles.
    parser.add_argument("--n_vehicles", type=int, default=10)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug_env", action="store_true", default=True,
                        help="Print actual vehicle counts after the first reset.")
    parser.add_argument("--no_debug_env", action="store_false", dest="debug_env",
                        help="Do not print environment debug information.")
    return parser.parse_args()


def main():
    args = parse_args()
    record_episodes(
        algo=args.algo,
        ckpt_path=args.ckpt,
        n_episodes=args.n_episodes,
        output_dir=args.output_dir,
        device=args.device,
        noise_std=args.noise_std,
        n_vehicles=args.n_vehicles,
        fps=args.fps,
        debug_env=args.debug_env,
    )


if __name__ == "__main__":
    main()
