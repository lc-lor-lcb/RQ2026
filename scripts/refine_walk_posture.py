"""Reproduce the high-body walk refinement from the saved forward-walking policy."""
import argparse
from dataclasses import asdict
import hashlib
import json
import shutil
from pathlib import Path

from roboquest.utils.reward_utils import WalkRewardConfig
from scripts.notebook_workflow import train_policy


def refine_posture(source, destination):
    """Keep the original policy; save matching checkpoint/statistics at each stage.

    Recipe qualified with walk_verified_forward (1,520,048 training steps).
    A different starting policy or platform requires a fresh evaluation.
    """
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f'Choose a new experiment directory: {destination}')
    required = ('walk_model.zip', 'walk_model_vecnorm.pkl', 'walk_params.json')
    for name in required:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    destination.mkdir(parents=True)
    for name in required:
        shutil.copy2(source / name, destination / name)
    original = json.loads((source / 'walk_params.json').read_text())
    reward = WalkRewardConfig(**original['reward_config'])
    reward.base_height_target = .26
    reward.base_height_weight = -500.
    reward.knee_height_target = .10
    reward.knee_height_weight = -100.
    reward.nonfoot_contact_weight = -1.
    reward.orientation_weight = -3.
    reward.vertical_velocity_weight = -.5
    history = {'source_sha256': hashlib.sha256((source/'walk_model.zip').read_bytes()).hexdigest(),
               'stages': []}

    def stage(name, steps, log_std, rate, kl, mode, exact_checkpoint=True,
              max_episode_steps=1000, n_steps=2048, batch_size=512):
        params = json.loads((destination / 'walk_params.json').read_text())
        env_kwargs = dict(params['walk_env_kwargs'], command_mode=mode,
                          max_episode_steps=max_episode_steps)
        train_policy('walk', destination, reward, steps, 4, params['ppo_kwargs'],
            resume=True, allow_reward_change=True, initial_log_std=log_std,
            checkpoint_steps=steps, walk_env_kwargs=env_kwargs,
            ppo_overrides={'learning_rate': rate, 'target_kl': kl,
                           'n_steps': n_steps, 'batch_size': batch_size})
        if exact_checkpoint:
            selected = params['total_timesteps'] + (steps // 4) * 4
            shutil.copy2(destination/'checkpoints'/f'walk_{selected}_steps.zip',
                         destination/'walk_model.zip')
            shutil.copy2(destination/'checkpoints'/f'walk_vecnormalize_{selected}_steps.pkl',
                         destination/'walk_model_vecnorm.pkl')
            saved = json.loads((destination/'walk_params.json').read_text())
            saved.update(total_timesteps=selected, selected_checkpoint=selected)
            (destination/'walk_params.json').write_text(json.dumps(saved, indent=2))
        history['stages'].append(dict(name=name, requested_steps=steps,
            reward=asdict(reward), log_std=log_std, learning_rate=rate, target_kl=kl,
            command_mode=mode, exact_checkpoint=exact_checkpoint,
            max_episode_steps=max_episode_steps, n_steps=n_steps, batch_size=batch_size))
        (destination/'walk_posture_curriculum.json').write_text(json.dumps(history, indent=2))

    stage('raise_body', 300_000, -.5, 3e-4, .01, 'uniform')
    reward.knee_height_weight = -1000.
    reward.nonfoot_contact_weight = -2.
    reward.orientation_weight = -5.
    reward.vertical_velocity_weight = 0.
    stage('stop_and_forward', 500_000, -1., 3e-4, .02, 'axis')
    reward.orientation_weight = -10.
    reward.vertical_velocity_weight = -1.
    stage('reduce_tilt', 300_000, -1.8, 3e-5, .003, 'axis', False)
    stage('practice_start', 200_000, -1.5, 1e-4, .01, 'axis',
          max_episode_steps=200, n_steps=512, batch_size=256)
    reward.lin_vel_weight = 6.
    reward.linear_tracking_variance = .16
    reward.ang_vel_weight = 3.
    stage('track_speed', 500_000, -2., 1e-4, .01, 'axis', False,
          n_steps=1024, batch_size=256)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('destination')
    args = parser.parse_args()
    refine_posture(args.source, args.destination)
