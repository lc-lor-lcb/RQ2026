"""Reproducible local walking run; checkpoint pairs and fixed-command evaluation.

Run from repository root: python -m scripts.tune_walk --steps 1000000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from roboquest.envs.go2_walk_env import Go2WalkEnv, CONTROL_DT
from roboquest.utils.reward_utils import WalkRewardConfig
from scripts.notebook_workflow import WALK_PPO, train_policy, train_walk_curriculum

COMMANDS = {'stand': (0, 0, 0), 'forward': (.4, 0, 0),
            'backward': (-.3, 0, 0), 'left': (0, .2, 0),
            'right': (0, -.2, 0), 'turn_left': (0, 0, .4),
            'turn_right': (0, 0, -.4)}


def evaluate(folder, seeds=(100, 101, 102), seconds=20, commands=None):
    folder = Path(folder)
    commands = COMMANDS if commands is None else commands
    params_path = folder / 'walk_params.json'
    env_kwargs = json.loads(params_path.read_text()).get('walk_env_kwargs', {}) if params_path.exists() else {}
    base = make_vec_env(lambda: Go2WalkEnv(randomize_cmd=False, **env_kwargs), n_envs=1)
    norm = VecNormalize.load(str(folder / 'walk_model_vecnorm.pkl'), base)
    norm.training = False
    norm.norm_reward = False
    model = PPO.load(folder / 'walk_model', device='cpu')
    raw = base.envs[0].unwrapped
    rows = []
    try:
        for name, command in commands.items():
            for seed in seeds:
                raw.set_vel_cmd(*command)
                obs, _ = raw.reset(seed=seed)
                velocities, tilts, heights, vertical_speeds = [], [], [], []
                nonfoot_contacts, hind_knee_heights = [], []
                all_nonfoot_contacts, all_hind_knee_heights = [], []
                joint_positions, joint_speeds, action_changes = [], [], []
                fell = False
                for step in range(round(seconds / CONTROL_DT)):
                    action, _ = model.predict(norm.normalize_obs(obs), deterministic=True)
                    action_change = action - raw._last_action
                    obs, _, terminated, truncated, _ = raw.step(action)
                    v = raw._world_to_body(raw.data.qvel[:3])
                    posture = raw.posture_metrics()
                    all_nonfoot_contacts.append(posture['nonfoot_contacts'] > 0)
                    all_hind_knee_heights.append(float(np.min(posture['knee_heights'][2:])))
                    if step >= 50:
                        joint_positions.append(raw.data.qpos[raw._act_qposadr].copy())
                        joint_speeds.append(raw.data.qvel[raw._act_dofadr].copy())
                        action_changes.append(action_change.copy())
                        nonfoot_contacts.append(posture['nonfoot_contacts'] > 0)
                        hind_knee_heights.append(float(np.min(posture['knee_heights'][2:])))
                        velocities.append([v[0], v[1], raw.data.qvel[5]])
                        heights.append(float(raw.data.qpos[2]))
                        vertical_speeds.append(float(v[2]))
                        tilts.append(float(np.arccos(np.clip(-raw._projected_gravity()[2], -1, 1))))
                    if terminated:
                        fell = True
                        break
                values = np.asarray(velocities) if velocities else np.full((1,3), np.nan)
                dominant_frequency = None
                if len(joint_positions) > 100 and np.sqrt(np.mean(np.square(joint_speeds))) > .1:
                    positions = np.asarray(joint_positions)
                    centered = positions - positions.mean(axis=0)
                    power = np.sum(np.abs(np.fft.rfft(centered * np.hanning(len(centered))[:, None], axis=0))**2, axis=1)
                    frequencies = np.fft.rfftfreq(len(centered), CONTROL_DT)
                    band = (frequencies >= .5) & (frequencies <= 12.)
                    dominant_frequency = float(frequencies[band][np.argmax(power[band])])
                rows.append(dict(command=name, seed=seed, fell=fell,
                    seconds=(step+1)*CONTROL_DT, target=list(command),
                    mean_velocity=np.mean(values, axis=0).tolist(),
                    velocity_rmse=np.sqrt(np.mean((values-command)**2, axis=0)).tolist(),
                    mean_tilt_rad=float(np.mean(tilts)) if tilts else None,
                    mean_height_m=float(np.mean(heights)) if heights else None,
                    min_height_m=float(np.min(heights)) if heights else None,
                    height_std_m=float(np.std(heights)) if heights else None,
                    nonfoot_contact_fraction=float(np.mean(nonfoot_contacts)) if nonfoot_contacts else None,
                    min_hind_knee_height_m=float(np.min(hind_knee_heights)) if hind_knee_heights else None,
                    whole_episode_nonfoot_contact_fraction=float(np.mean(all_nonfoot_contacts)),
                    whole_episode_min_hind_knee_height_m=float(np.min(all_hind_knee_heights)),
                    joint_speed_rms_rad_s=float(np.sqrt(np.mean(np.square(joint_speeds)))) if joint_speeds else None,
                    action_change_rms=float(np.sqrt(np.mean(np.square(action_changes)))) if action_changes else None,
                    dominant_joint_frequency_hz=dominant_frequency,
                    vertical_speed_rms=float(np.sqrt(np.mean(np.square(vertical_speeds)))) if vertical_speeds else None))
        for row in rows:
            target = np.asarray(row['target'])
            tolerance = np.maximum(np.abs(target) * .35, [.06, .06, .12])
            row['passed'] = bool(not row['fell'] and row['mean_tilt_rad'] is not None
                and row['mean_tilt_rad'] < .35
                and np.all(np.abs(np.asarray(row['mean_velocity']) - target) <= tolerance)
                and np.all(np.asarray(row['velocity_rmse']) <= [.2, .15, .3]))
            row['posture_passed'] = bool(row['passed']
                and row['mean_height_m'] >= .24 and row['min_height_m'] >= .22
                and row['height_std_m'] <= .02
                and row['nonfoot_contact_fraction'] == 0
                and row['min_hind_knee_height_m'] >= .06)
            row['startup_passed'] = bool(row['whole_episode_nonfoot_contact_fraction'] == 0
                and row['whole_episode_min_hind_knee_height_m'] >= .06)
        report = dict(rows=rows, fall_rate=sum(r['fell'] for r in rows)/len(rows),
                      passed=all(r['passed'] for r in rows),
                      posture_passed=all(r['posture_passed'] for r in rows),
                      startup_passed=all(r['startup_passed'] for r in rows),
                      posture_criteria={'mean_height_min_m': .24, 'min_height_m': .22,
                        'height_std_max_m': .02, 'nonfoot_contact_fraction_max': 0,
                        'hind_knee_height_min_m': .06, 'warmup_seconds': 1,
                        'startup': 'separately checks contacts and knee height including first second'},
                      criteria={'duration_seconds': seconds, 'warmup_seconds': 1,
                        'mean_error': 'max(35% of target, [0.06 m/s, 0.06 m/s, 0.12 rad/s])',
                        'rmse_max': [.2, .15, .3], 'mean_tilt_max_rad': .35})
        def json_safe(value):
            if isinstance(value, dict):
                return {key: json_safe(item) for key, item in value.items()}
            if isinstance(value, list):
                return [json_safe(item) for item in value]
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value
        report = json_safe(report)
        (folder/'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        for name in commands:
            selected = [r for r in rows if r['command']==name]
            print(name, 'falls', sum(r['fell'] for r in selected), 'velocity',
                  np.round(np.mean([r['mean_velocity'] for r in selected], axis=0),3), flush=True)
        return report
    finally:
        norm.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000000)
    parser.add_argument('--curriculum', action='store_true', help='Run the staged forward-walking recipe; --steps is stage 1')
    parser.add_argument('--folder', default='models/teams/local_walk')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--gait-candidate', action='store_true')
    parser.add_argument('--forward-curriculum', action='store_true')
    parser.add_argument('--action-scale', type=float, default=.3)
    parser.add_argument('--bootstrap-gait', action='store_true')
    parser.add_argument('--expand-commands', action='store_true')
    parser.add_argument('--steering-curriculum', action='store_true')
    parser.add_argument('--refine', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    folder = Path(args.folder)
    folder.mkdir(parents=True, exist_ok=True)
    if args.eval_only:
        evaluate(folder)
        return
    if args.curriculum:
        if args.resume:
            parser.error('--curriculum starts a fresh run; use --resume alone to continue the last stage')
        train_walk_curriculum(folder, first_stage_steps=args.steps)
        evaluate(folder)
        return
    # No invisible clock target: the 45D observation contains no gait phase.
    reward = WalkRewardConfig(feet_gait_weight=0, action_rate_weight=-.01)
    if args.gait_candidate:
        reward = WalkRewardConfig(lin_vel_weight=3., ang_vel_weight=1.,
            feet_gait_weight=0., diagonal_support_weight=.5,
            action_rate_weight=-.002, foot_slip_weight=-.02)
    settings = dict(WALK_PPO, device='cpu', ent_coef=.0)
    env_kwargs = {}
    if args.forward_curriculum:
        reward = WalkRewardConfig(lin_vel_weight=3., ang_vel_weight=1.,
            feet_gait_weight=0., diagonal_support_weight=.5,
            action_rate_weight=-.001, foot_slip_weight=-.02)
        settings.update(target_kl=.02, batch_size=512)
        env_kwargs = {'command_ranges': {'vx': [.2, .5], 'vy': [0, 0], 'omega': [0, 0]}}
    env_kwargs['action_scale'] = args.action_scale
    if args.bootstrap_gait:
        if args.action_scale != .6 or not args.forward_curriculum:
            parser.error('Gait examples currently require --forward-curriculum --action-scale .6')
        settings.update(learning_rate=1e-4, n_epochs=5, target_kl=.01)
    if args.resume:
        params = json.loads((folder/'walk_params.json').read_text())
        reward = WalkRewardConfig(**params['reward_config'])
        settings = params['ppo_kwargs']
        env_kwargs = params.get('walk_env_kwargs', {})
    if args.expand_commands:
        env_kwargs.update(command_ranges={'vx': [-.4, .5], 'vy': [-.25, .25], 'omega': [-.5, .5]},
                          command_mode='axis')
    if args.steering_curriculum:
        env_kwargs.update(command_ranges={'vx': [.2, .5], 'vy': [0, 0], 'omega': [-.5, .5]},
                          command_mode='uniform')
    if args.refine:
        reward.linear_tracking_variance = .09
        reward.angular_tracking_variance = .16
        reward.base_height_weight = -30.
        reward.vertical_velocity_weight = -.5
    train_policy('walk', folder, reward, args.steps, 4, settings,
                 resume=args.resume, verbose=1, progress_bar=False, walk_env_kwargs=env_kwargs, bootstrap_gait=args.bootstrap_gait,
                 allow_reward_change=args.refine, initial_log_std=-1. if args.refine else None)
    evaluate(folder)


if __name__ == '__main__':
    main()
