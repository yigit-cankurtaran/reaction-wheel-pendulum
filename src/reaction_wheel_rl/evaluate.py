"""Evaluate a trained model, report task metrics, and optionally render a GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import SAC

from reaction_wheel_rl.env import ReactionWheelPendulumEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/best_model.zip"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--gif", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = ReactionWheelPendulumEnv(render_mode="rgb_array" if args.gif else None)
    model = SAC.load(args.model)
    returns: list[float] = []
    balanced_fractions: list[float] = []
    frames: list[np.ndarray] = []

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        done = False
        total_reward = 0.0
        balanced_steps = 0
        steps = 0
        if args.gif and episode == 0:
            frames.append(env.render())
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            balanced_steps += int(info["is_balanced"])
            steps += 1
            if args.gif and episode == 0:
                frames.append(env.render())
        returns.append(total_reward)
        balanced_fractions.append(balanced_steps / steps)

    print(f"Return: {np.mean(returns):.1f} +/- {np.std(returns):.1f}")
    print(f"Balanced fraction: {100 * np.mean(balanced_fractions):.1f}%")
    if args.gif:
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.gif, frames, duration=1 / env.metadata["render_fps"], loop=0)
        print(f"Saved {args.gif}")
    env.close()


if __name__ == "__main__":
    main()

