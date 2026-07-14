# Reaction-Wheel Pendulum RL

This project trains a MuJoCo robot to recover from attitude disturbances and balance
using **only an internal reaction wheel**. Applying motor torque accelerates the orange
flywheel and produces
an equal-and-opposite torque on the blue body—the same principle used for spacecraft
attitude control and self-righting robots.

The custom Gymnasium environment observes
`[cos(body angle), sin(body angle), body angular velocity, wheel angular velocity]`
with angular rates scaled to neural-network-friendly ranges, and commands one bounded
wheel torque. SAC learns recovery and fine stabilization from a dense uprightness
reward. Episodes usually begin within 35 degrees of upright, with occasional global
starts; full hanging-down swing-up is an intentionally harder extension.

## Setup

```bash
UV_CACHE_DIR=.uv-cache uv sync
```

This creates `.venv` with Python 3.12 and installs MuJoCo, Gymnasium, PyTorch, and
Stable-Baselines3 from `uv.lock`.

## Train and evaluate

```bash
UV_CACHE_DIR=.uv-cache uv run python -m reaction_wheel_rl.train --steps 40000
MUJOCO_GL=osmesa UV_CACHE_DIR=.uv-cache uv run python -m reaction_wheel_rl.evaluate \
  --model artifacts/best_model.zip --gif artifacts/swingup.gif
```

On macOS, rendering normally works without setting `MUJOCO_GL`; use the default
command below:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m reaction_wheel_rl.evaluate \
  --model artifacts/best_model.zip --gif artifacts/swingup.gif
```

The evaluator reports average return and the percentage of time the body is within
12 degrees of upright with low angular velocity. To inspect the environment:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest
```

To watch a trained policy in a MuJoCo window, continuously across episodes, run:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m reaction_wheel_rl.watch \
  --model artifacts/best_model.zip
```

It runs at real time by default. Close the window or press `Ctrl-C` to stop; pass
`--speed 2` for double-speed playback.

## Ideas to extend it

- Randomize mass, friction, motor strength, and sensor noise for sim-to-real work.
- Limit wheel speed, forcing the policy to learn momentum dumping.
- Expand the reset distribution into a full hanging-down swing-up curriculum.
- Add a second body axis and two reaction wheels, then progress to a free 3-D cube.
- Compare SAC against an energy-shaping controller or model-predictive control.
