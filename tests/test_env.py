import numpy as np
from gymnasium.utils.env_checker import check_env

from reaction_wheel_rl import ReactionWheelPendulumEnv


def test_gymnasium_contract() -> None:
    env = ReactionWheelPendulumEnv()
    check_env(env, skip_render_check=True)
    env.close()


def test_rollout_is_finite_and_truncates() -> None:
    env = ReactionWheelPendulumEnv(max_episode_steps=12)
    obs, _ = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    for step in range(12):
        obs, reward, terminated, truncated, info = env.step(np.array([0.2], dtype=np.float32))
        assert np.isfinite(reward)
        assert env.observation_space.contains(obs)
        assert "angle_error_rad" in info
        assert not terminated
    assert step == 11 and truncated
    env.close()


def test_upright_state_has_high_reward() -> None:
    env = ReactionWheelPendulumEnv()
    env.reset(seed=1, options={"upright": True})
    _, reward, _, _, info = env.step(np.zeros(1, dtype=np.float32))
    assert reward > 3.0
    assert info["angle_error_rad"] < 0.2
    env.close()

