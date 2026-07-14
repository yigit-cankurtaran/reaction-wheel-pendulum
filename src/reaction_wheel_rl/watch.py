"""Watch a trained reaction-wheel policy in a real-time MuJoCo window."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import SAC

from reaction_wheel_rl.env import ReactionWheelPendulumEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/best_model.zip"))
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: real time).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.speed <= 0:
        raise ValueError("--speed must be positive")

    env = ReactionWheelPendulumEnv(render_mode="human")
    model = SAC.load(args.model)
    episode = 0
    print("Watching policy. Close the MuJoCo window or press Ctrl-C to stop.")

    try:
        while env.viewer_is_running:
            obs, _ = env.reset(seed=args.seed + episode)
            env.render()
            if not env.viewer_is_running:
                break
            episode_return = 0.0
            for _ in range(env.max_episode_steps):
                started = time.perf_counter()
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += reward

                remaining = env.dt / args.speed - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)
                if terminated or truncated or not env.viewer_is_running:
                    break

            if not env.viewer_is_running:
                break
            episode += 1
            print(f"Episode {episode}: return {episode_return:.1f}")
    except KeyboardInterrupt:
        print("\nStopped watching.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
