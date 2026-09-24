"""Learn a slow 1.5 Hz gait example, then refine it with PPO in the same physics.

The deployed policy remains a 45-input neural policy: the example's clock is
used only to generate training examples, never to drive policy evaluation.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from tqdm.auto import trange
from roboquest.envs.go2_walk_env import Go2WalkEnv, STANDING_POS
from roboquest.utils.reward_utils import WalkRewardConfig
from scripts.notebook_workflow import train_policy

SMOOTH_STEPS = 10_000
SMOOTH_PPO = dict(learning_rate=1e-6, n_steps=1024, batch_size=256, n_epochs=5,
    gamma=.99, gae_lambda=.95, ent_coef=0., target_kl=.001,
    policy_kwargs={'net_arch': [256, 256, 128]}, device='cpu')

SMOOTH_ENV = dict(action_scale=.6, joint_stiffness=60., joint_damping=2.,
    torque_limits=[23.7, 23.7, 45.43], command_mode='axis',
    command_ranges={'vx': [.4, .4], 'vy': [0., 0.], 'omega': [0., 0.]})
SMOOTH_REWARD = dict(linear_tracking_variance=.16, angular_tracking_variance=.04,
    lin_vel_weight=6., ang_vel_weight=3., orientation_weight=-10.,
    base_height_target=.29, base_height_weight=-500., vertical_velocity_weight=-2.,
    nonfoot_contact_weight=-2., knee_height_target=.10, knee_height_weight=-1000.,
    action_rate_weight=-.05, joint_velocity_weight=-.003,
    stand_joint_velocity_weight=-.015, feet_gait_weight=0.,
    diagonal_support_weight=0., foot_slip_weight=-.02)


def example_action(time, vx, frequency=1.5, height=.31, lift=.05):
    """Two-link inverse kinematics, with alternating diagonal support."""
    if abs(vx) < .05:
        x, z = np.zeros(4), np.full(4, -height)
    else:
        phase = (time * frequency + np.array([0., .5, .5, 0.])) % 1
        swing = phase < .5
        u = np.where(swing, phase * 2, (phase - .5) * 2)
        extent = 1.6 * vx / (frequency * 4)
        x = np.where(swing, -extent * np.cos(np.pi * u), extent * (1 - 2*u))
        z = -height + np.where(swing, lift * np.sin(np.pi * u), 0.)
    length = .213
    knee = -np.arccos(np.clip((x*x + z*z - 2*length**2)/(2*length**2), -1, 1))
    thigh = np.arctan2(-x, -z) - knee/2
    target = np.stack([np.zeros(4), thigh, knee], axis=1).reshape(-1)
    return np.clip((target - STANDING_POS)/.6, -1, 1).astype(np.float32)


def train_smooth_walk(folder, steps=SMOOTH_STEPS, seed=0, sample_steps=30_000, updates=4000,
                      reward_config=None, ppo_kwargs=None, num_envs=4, progress_bar=True):
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(folder)
    folder.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    from copy import deepcopy
    reward = reward_config or WalkRewardConfig(**SMOOTH_REWARD)
    settings = deepcopy(SMOOTH_PPO)
    settings.update(ppo_kwargs or {})
    env = VecNormalize(make_vec_env(lambda: Go2WalkEnv(**SMOOTH_ENV), n_envs=num_envs, seed=seed))
    model = PPO('MlpPolicy', env, seed=seed, **settings)
    teacher = Go2WalkEnv(randomize_cmd=False, reward_config=reward, **SMOOTH_ENV)
    observations, targets, returns = [], [], []
    sample_iter = trange(sample_steps, desc="walk examples", unit="step", disable=not progress_bar)
    for step in sample_iter:
        local_step = step % 500
        if local_step == 0:
            discounted_return = 0.
            command = 0. if rng.random() < .35 else float(rng.uniform(.3, .45))
            teacher.set_vel_cmd(command, 0., 0.)
            obs, _ = teacher.reset(seed=seed+step)
        if local_step == 250:
            # Teach starting from a settled stance as well as stopping mid-gait.
            # A policy trained only on resets can otherwise remain motionless
            # when a user moves the velocity slider after standing still.
            command = float(rng.uniform(.3, .45)) if command == 0. else 0.
            teacher.set_vel_cmd(command, 0., 0.)
            obs = teacher._get_obs().astype(np.float32)
        action = example_action((local_step % 250)*.02, command)
        observations.append(obs.copy())
        targets.append(action)
        obs, r, terminated, _, _ = teacher.step(action)
        discounted_return = .99 * discounted_return + r
        returns.append(discounted_return)
        if terminated:
            raise RuntimeError('Gait example fell; do not use failed examples')
    teacher.close()
    observations = np.asarray(observations, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    env.obs_rms.update(observations)
    env.ret_rms.update(np.asarray(returns))
    # Off-trajectory observations make the learned cycle tolerate small errors.
    augmented = observations.copy()
    augmented[:, 3:33] += rng.normal(0, .05, augmented[:, 3:33].shape)
    augmented[:, -12:] += rng.normal(0, .015, augmented[:, -12:].shape)
    x = torch.as_tensor(env.normalize_obs(np.concatenate([observations, augmented])), device='cpu')
    y = torch.as_tensor(np.concatenate([targets, targets]), device='cpu')
    actor = list(model.policy.mlp_extractor.policy_net.parameters()) + list(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(actor, lr=3e-4)
    update_iter = trange(updates, desc="walk imitation", unit="update", disable=not progress_bar)
    for update in update_iter:
        indices = torch.as_tensor(rng.integers(0, len(x), 512))
        prediction = model.policy._predict(x[indices], deterministic=True)
        loss = (prediction-y[indices]).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if progress_bar:
            update_iter.set_postfix(mse=f"{float(loss.item()):.3g}")
        elif (update+1) % 1000 == 0:
            print('example update', update+1, 'MSE', float(loss.item()), flush=True)
    with torch.no_grad():
        model.policy.log_std.fill_(-3.)
    model.policy.optimizer.state.clear()
    model.save(folder/'walk_model')
    env.save(str(folder/'walk_model_vecnorm.pkl'))
    params = dict(kind='walk', reward_config=asdict(reward), ppo_kwargs=settings,
        total_timesteps=0, control_dt=.02, environment_revision='walk-v2',
        walk_env_kwargs=SMOOTH_ENV, seed=seed,
        example_training={'sample_steps':sample_steps, 'updates':updates, 'mse':float(loss.item())})
    (folder/'walk_params.json').write_text(json.dumps(params, indent=2))
    import shutil
    before = folder/'examples_only'; before.mkdir()
    for name in ('walk_model.zip', 'walk_model_vecnorm.pkl', 'walk_params.json'):
        shutil.copy2(folder/name, before/name)
    env.close()
    if steps:
        train_policy('walk', folder, reward, steps, num_envs, settings, seed=seed, resume=True,
                     checkpoint_steps=10_000, progress_bar=progress_bar, freeze_normalization=True)
    (folder/'walk_smooth_curriculum.json').write_text(json.dumps(
        {'method':'inverse-kinematics examples then PPO', 'examples':params['example_training'],
         'requested_ppo_steps':steps, 'seed':seed, 'freeze_normalization': True,
         'command_transition_examples': True,
         'actual_ppo_steps': json.loads((folder/'walk_params.json').read_text())['total_timesteps'],
         'recipe': 'scripts.bootstrap_smooth_walk.train_smooth_walk'}, indent=2))
    return folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder')
    parser.add_argument('--steps', type=int, default=SMOOTH_STEPS)
    args = parser.parse_args()
    train_smooth_walk(args.folder, args.steps)
