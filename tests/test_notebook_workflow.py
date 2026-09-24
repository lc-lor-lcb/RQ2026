"""Smoke-test the shared workflow against the real MuJoCo environments."""
from pathlib import Path
import sys

import pytest
import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.notebook_workflow import train_policy, evaluate_flee
from roboquest.utils.reward_utils import WalkRewardConfig, FleeRewardConfig


def test_saved_hierarchy_can_be_evaluated_without_retraining(tmp_path):
    settings = dict(n_steps=8, batch_size=8, n_epochs=1,
                    policy_kwargs={'net_arch': [8]}, device='cpu')
    train_policy('walk', tmp_path, WalkRewardConfig(), 8, 1, settings,
                 walk_env_kwargs={'action_scale': .6})
    train_policy('flee', tmp_path, FleeRewardConfig(), 8, 1, settings)
    for kind, obs_size, action_size in [('walk', 45, 12), ('flee', 10, 3)]:
        model = PPO.load(tmp_path / f'{kind}_model')
        assert model.observation_space.shape == (obs_size,)
        assert model.action_space.shape == (action_size,)
        assert (tmp_path / f'{kind}_params.json').is_file()
    from roboquest.envs.go2_tag_hierarchical_env import Go2TagHierarchicalEnv
    hierarchical = Go2TagHierarchicalEnv(str(tmp_path / 'walk_model'))
    assert hierarchical._low_env.action_scale == .6
    hierarchical.close()
    before = {p.name: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    rows = evaluate_flee(tmp_path, seeds=(100, 101))
    assert [row['seed'] for row in rows] == [100, 101]
    for row in rows:
        assert 0 < row['survived_seconds'] <= 60
        assert np.isfinite(row['mean_distance'])
        assert sum(row[k] for k in ['escaped', 'tagged', 'fell']) == 1
    assert before == {p.name: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    repeat = evaluate_flee(tmp_path, seeds=(100, 101))
    assert rows == repeat


def test_evaluation_requires_walk_normalization(tmp_path):
    (tmp_path / 'walk_model.zip').touch()
    with pytest.raises(FileNotFoundError, match='walk_model_vecnorm.pkl'):
        evaluate_flee(tmp_path)


def test_resume_preserves_physics_and_export_contract(tmp_path):
    import json
    from scripts.preview_saved_walk import preview_saved_walk
    from scripts.build_mjswan_viewer import _walk_action_scale
    settings = dict(n_steps=8, batch_size=8, n_epochs=1,
                    policy_kwargs={'net_arch': [8]}, device='cpu')
    train_policy('walk', tmp_path, WalkRewardConfig(), 8, 1, settings,
                 walk_env_kwargs={'action_scale': .6}, checkpoint_steps=8)
    train_policy('walk', tmp_path, WalkRewardConfig(), 8, 1, settings, resume=True)
    params = json.loads((tmp_path/'walk_params.json').read_text())
    assert params['total_timesteps'] == 16
    assert params['walk_env_kwargs']['action_scale'] == .6
    assert (tmp_path/'checkpoints/walk_8_steps.zip').is_file()
    assert (tmp_path/'checkpoints/walk_vecnormalize_8_steps.pkl').is_file()
    exported, has_stats = preview_saved_walk(tmp_path/'walk_model.zip')
    assert has_stats and _walk_action_scale(exported) == .6


def test_frozen_normalization_survives_resume(tmp_path):
    import pickle
    import json
    settings = dict(n_steps=8, batch_size=8, n_epochs=1,
                    policy_kwargs={'net_arch': [8]}, device='cpu')
    reward = WalkRewardConfig()
    train_policy('walk', tmp_path, reward, 8, 1, settings, progress_bar=False)
    path = tmp_path/'walk_model_vecnorm.pkl'
    before = pickle.loads(path.read_bytes())
    train_policy('walk', tmp_path, reward, 8, 1, settings, resume=True,
                 freeze_normalization=True, progress_bar=False)
    train_policy('walk', tmp_path, reward, 8, 1, settings, resume=True, progress_bar=False)
    after = pickle.loads(path.read_bytes())
    np.testing.assert_array_equal(before.obs_rms.mean, after.obs_rms.mean)
    np.testing.assert_array_equal(before.obs_rms.var, after.obs_rms.var)
    assert before.obs_rms.count == after.obs_rms.count
    assert before.ret_rms.count == after.ret_rms.count
    assert json.loads((tmp_path/'walk_params.json').read_text())['freeze_normalization']
