"""Regression checks for the physical policy input/output contract."""
import numpy as np
import pytest
from roboquest.envs.go2_walk_env import Go2WalkEnv, CONTROL_DT


def test_body_frame_rotation():
    env = Go2WalkEnv(randomize_cmd=False)
    try:
        env.reset(seed=0)
        env.data.qpos[3:7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
        np.testing.assert_allclose(env._world_to_body(np.array([0., 1., 0.])), [1, 0, 0], atol=1e-12)
        env.data.qpos[3:7] = [np.sqrt(.5), np.sqrt(.5), 0, 0]
        np.testing.assert_allclose(env._projected_gravity(), [0, -1, 0], atol=1e-12)
    finally:
        env.close()


def test_next_observation_contains_applied_action_and_50hz_time():
    env = Go2WalkEnv(randomize_cmd=False)
    try:
        env.reset(seed=0)
        action = np.linspace(-.2, .2, 12)
        obs, *_ = env.step(action)
        np.testing.assert_allclose(obs[-12:], action, atol=1e-7)
        assert env.data.time == pytest.approx(CONTROL_DT)
    finally:
        env.close()


def test_viewer_scene_has_same_dynamics():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    train = Go2WalkEnv(randomize_cmd=False)
    viewer = Go2WalkEnv(randomize_cmd=False, xml_path=str(root / 'models/go2/walk_web.xml'))
    try:
        train.reset(seed=42)
        viewer.reset(seed=42)
        rng = np.random.default_rng(5)
        for _ in range(100):
            action = rng.uniform(-.5, .5, 12)
            train.step(action)
            viewer.step(action)
        np.testing.assert_allclose(train.data.qpos, viewer.data.qpos, atol=1e-12)
        np.testing.assert_allclose(train.data.qvel, viewer.data.qvel, atol=1e-12)
    finally:
        train.close()
        viewer.close()


def test_posture_distinguishes_feet_from_collapsed_leg_contacts():
    import mujoco
    from roboquest.utils.reward_utils import WalkRewardConfig
    env = Go2WalkEnv(randomize_cmd=False)
    try:
        env.reset(seed=0)
        for _ in range(100):
            env.step(np.zeros(12))
        upright = env.posture_metrics()
        assert upright['nonfoot_contacts'] == 0
        env.data.qpos[2] = .08
        mujoco.mj_forward(env.model, env.data)
        collapsed = env.posture_metrics()
        assert collapsed['nonfoot_contacts'] > 0
        assert np.min(collapsed['knee_heights']) < .06
        action = np.zeros(12)
        before = env._compute_reward(action)
        env.reward_config.nonfoot_contact_weight = -1.
        env.reward_config.knee_height_weight = -100.
        assert env._compute_reward(action) < before
    finally:
        env.close()


def test_joint_motion_penalty_is_stronger_when_standing():
    from roboquest.utils.reward_utils import WalkRewardConfig
    env = Go2WalkEnv(randomize_cmd=False, reward_config=WalkRewardConfig(
        feet_gait_weight=0, diagonal_support_weight=0))
    try:
        env.reset(seed=0)
        env.data.qvel[env._act_dofadr] = 2.
        action = np.zeros(12)
        for command, expected_penalty in [(0., -.03 * 12 * 4), (.4, -.01 * 12 * 4)]:
            env.set_vel_cmd(command, 0, 0)
            env.reward_config.joint_velocity_weight = 0.
            env.reward_config.stand_joint_velocity_weight = 0.
            before = env._compute_reward(action)
            env.reward_config.joint_velocity_weight = -.01
            env.reward_config.stand_joint_velocity_weight = -.02
            assert env._compute_reward(action) - before == pytest.approx(expected_penalty)
    finally:
        env.close()


def test_saved_gains_and_limits_match_viewer_physics(tmp_path):
    import json
    import mujoco
    from pathlib import Path
    from scripts.build_mjswan_viewer import _configure_walk_scene
    params = dict(joint_stiffness=60., joint_damping=2., torque_limits=[23.7, 23.7, 45.43])
    (tmp_path/'walk_policy_contract.json').write_text(json.dumps(params))
    root = Path(__file__).resolve().parents[1]
    spec = mujoco.MjSpec.from_file(str(root/'models/go2/walk_web.xml'))
    _configure_walk_scene(spec, tmp_path/'walk_policy_normalized.onnx')
    model = spec.compile()
    env = Go2WalkEnv(randomize_cmd=False, **params)
    try:
        np.testing.assert_array_equal(model.actuator_gainprm, env.model.actuator_gainprm)
        np.testing.assert_array_equal(model.actuator_biasprm, env.model.actuator_biasprm)
        np.testing.assert_array_equal(model.dof_damping, env.model.dof_damping)
        np.testing.assert_array_equal(model.actuator_forcerange, env.model.actuator_forcerange)
        np.testing.assert_array_equal(model.actuator_forcelimited, env.model.actuator_forcelimited)
        env.reset(seed=3)
        data = mujoco.MjData(model)
        data.qpos[:] = env.data.qpos
        data.ctrl[:] = env.data.ctrl
        mujoco.mj_forward(model, data)
        rng = np.random.default_rng(5)
        for _ in range(100):
            env.step(rng.uniform(-1, 1, 12))
            data.ctrl[:] = env.data.ctrl
            for _ in range(env.frame_skip):
                mujoco.mj_step(model, data)
        np.testing.assert_allclose(data.qpos, env.data.qpos, atol=1e-10)
        assert np.all(np.abs(env.data.actuator_force) <= np.tile(params['torque_limits'], 4) + 1e-8)
    finally:
        env.close()
