from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import onnxruntime as ort
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.preview_saved_walk import preview_saved_walk


class StubWalk(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (45,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (12,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.ones(45, dtype=np.float32), {}

    def step(self, action):
        return np.ones(45, dtype=np.float32), 0., False, False, {}


@pytest.mark.parametrize('with_stats', [True, False])
def test_saved_policy_exports_without_retraining(tmp_path, with_stats, capsys):
    env = VecNormalize(DummyVecEnv([StubWalk]))
    env.reset()
    model = PPO('MlpPolicy', env, n_steps=8, batch_size=8, device='cpu')
    source = tmp_path / 'walk_model.zip'
    model.save(source)
    before = source.read_bytes()
    if with_stats:
        env.save(tmp_path / 'walk_model_vecnorm.pkl')
    output, found = preview_saved_walk(source)
    assert found == with_stats
    assert source.read_bytes() == before
    raw_obs = np.full((1, 45), 0.5, dtype=np.float32)
    obs = env.normalize_obs(raw_obs) if with_stats else raw_obs
    expected, _ = model.predict(obs, deterministic=True)
    session = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
    actual = session.run(None, {'obs': raw_obs})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    if not with_stats:
        assert '正規化データがありません' in capsys.readouterr().out
    env.close()


def test_explicit_missing_stats_does_not_silently_fall_back(tmp_path):
    model = PPO('MlpPolicy', StubWalk(), n_steps=8, batch_size=8, device='cpu')
    model.save(tmp_path / 'walkmodel.zip')
    with pytest.raises(FileNotFoundError, match='正規化データ'):
        preview_saved_walk(tmp_path / 'walkmodel.zip', tmp_path / 'typo.pkl')
