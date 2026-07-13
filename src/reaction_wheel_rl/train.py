"""Train a Soft Actor-Critic policy on the reaction-wheel balancer."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from reaction_wheel_rl.env import ReactionWheelPendulumEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)
    train_env = Monitor(ReactionWheelPendulumEnv())
    eval_env = Monitor(ReactionWheelPendulumEnv())

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=150_000,
        learning_starts=2_000,
        batch_size=256,
        tau=0.01,
        gamma=0.99,
        ent_coef=0.05,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": [128, 128]},
        seed=args.seed,
        verbose=1,
        tensorboard_log=None,
        device="auto",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.output),
        log_path=str(args.output),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )
    checkpoint = CheckpointCallback(
        save_freq=25_000, save_path=str(args.output / "checkpoints"), name_prefix="sac"
    )
    model.learn(total_timesteps=args.steps, callback=[eval_callback, checkpoint], progress_bar=False)
    model.save(args.output / "final_model")
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"Evaluation return: {mean:.1f} +/- {std:.1f}")
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
