"""Shared training, persistence and evaluation for the Japanese notebooks."""
from dataclasses import asdict
from pathlib import Path
import json
import platform
import time
from importlib.metadata import version
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from roboquest.envs.go2_walk_env import Go2WalkEnv, CONTROL_DT
from roboquest.envs.go2_tag_hierarchical_env import Go2TagHierarchicalEnv
from roboquest.utils.reward_utils import FleeRewardConfig

WALK_PPO = dict(learning_rate=3e-4, n_steps=2048, batch_size=256,
                n_epochs=10, gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
                policy_kwargs={'net_arch': [256, 256, 128]})
FLEE_PPO = dict(learning_rate=1e-4, n_steps=1024, batch_size=128,
                n_epochs=10, gamma=0.99, gae_lambda=0.95, ent_coef=0.02,
                policy_kwargs={'net_arch': [128, 128]})


# Forward-motion experiment settings; all-direction qualification is still pending.
WALK_FORWARD_PPO = dict(WALK_PPO, batch_size=512, ent_coef=0.0, target_kl=.02, device='cpu')
WALK_FORWARD_REWARD = dict(lin_vel_weight=3., ang_vel_weight=1., orientation_weight=-1.,
    torques_weight=-2.5e-5, action_rate_weight=-.001, feet_gait_weight=0.,
    diagonal_support_weight=.5, foot_slip_weight=-.02)
WALK_FORWARD_ENV = dict(action_scale=.6,
    command_ranges={'vx': [.2, .5], 'vy': [0., 0.], 'omega': [0., 0.]})
WALK_FORWARD_STEPS = 750_000


def load_bundled_walk(save_dir, bundle_dir='models/teams/bundled_walk'):
    """Copy an explicitly selected local bundle into the notebook experiment folder."""
    import shutil
    source, destination = Path(bundle_dir), Path(save_dir)
    names = ['walk_model.zip', 'walk_model_vecnorm.pkl', 'walk_params.json', 'evaluation.json']
    for name in names:
        if not (source/name).is_file():
            raise FileNotFoundError(source/name)
    destination.mkdir(parents=True, exist_ok=True)
    names += [name for name in ('walk_curriculum.json', 'walk_heldout_evaluation.json',
                               'walk_posture_curriculum.json', 'walk_smooth_curriculum.json',
                               'walk_transition_evaluation.json') if (source/name).is_file()]
    for name in names:
        if (source/name).resolve() != (destination/name).resolve():
            shutil.copy2(source/name, destination/name)
    print('ローカルで保存したモデル・正規化データ・設定を読み込みました。')


def train_policy(kind, save_dir, reward_config, timesteps, num_envs,
                 ppo_kwargs, seed=0, oni_speed=0.025, resume=False,
                 checkpoint_steps=250_000, verbose=0, progress_bar=True, walk_env_kwargs=None, bootstrap_gait=False,
                 allow_reward_change=False, initial_log_std=None, ppo_overrides=None,
                 freeze_normalization=None):
    """Train or resume PPO; save its matching normalization, physics and settings."""
    torch.set_num_threads(1)
    folder = Path(save_dir)
    folder.mkdir(parents=True, exist_ok=True)
    if kind == 'walk':
        factory = lambda: Go2WalkEnv(reward_config=reward_config,
                                    **dict({'max_episode_steps': 1000}, **(walk_env_kwargs or {})))
    elif kind == 'flee':
        factory = lambda: Go2TagHierarchicalEnv(
            low_level_model_path=str(folder / 'walk_model'),
            low_level_vecnorm_path=str(folder / 'walk_model_vecnorm.pkl'),
            flee_config=reward_config, oni_speed=oni_speed)
        for name in ('walk_model.zip', 'walk_model_vecnorm.pkl'):
            if not (folder / name).is_file():
                raise FileNotFoundError(f'歩行モデルと正規化データが必要です: {folder / name}')
    else:
        raise ValueError(f'Unknown policy kind: {kind}')
    if resume:
        previous = json.loads((folder / f'{kind}_params.json').read_text())
        if walk_env_kwargs is None:
            walk_env_kwargs = previous.get('walk_env_kwargs', {})
        if previous.get('environment_revision') != 'walk-v2':
            raise ValueError('旧環境のモデルは継続せず、修正後の環境で新規学習してください。')
        if not allow_reward_change and asdict(type(reward_config)(**previous['reward_config'])) != asdict(reward_config):
            raise ValueError('継続時の報酬設定は保存済み設定と一致させてください。')
    started = time.monotonic()
    vec_env_cls = SubprocVecEnv if num_envs > 1 else None
    base = make_vec_env(factory, n_envs=num_envs, seed=seed, vec_env_cls=vec_env_cls)
    if resume:
        try:
            env = VecNormalize.load(str(folder / f'{kind}_model_vecnorm.pkl'), base)
        except Exception:
            base.close()
            raise
    else:
        env = VecNormalize(base, norm_obs=True, norm_reward=True)
    try:
        if freeze_normalization is None:
            freeze_normalization = previous.get('freeze_normalization', False) if resume else False
        env.training = not freeze_normalization
        settings = dict(previous['ppo_kwargs'] if resume else ppo_kwargs)
        settings.update(ppo_overrides or {})
        settings.setdefault('device', 'cpu')
        if resume:
            model = PPO.load(str(folder / f'{kind}_model'), env=env, **settings)
        else:
            model = PPO('MlpPolicy', env, seed=seed, verbose=verbose, **settings)
        if initial_log_std is not None:
            with torch.no_grad():
                model.policy.log_std.fill_(initial_log_std)
        bootstrap_result = None
        if bootstrap_gait and not resume:
            from scripts.bootstrap_walk import pretrain_examples
            bootstrap_result = pretrain_examples(model, env, seed=seed)
            initial = folder / 'bootstrap'
            initial.mkdir(exist_ok=True)
            model.save(initial / 'walk_model')
            env.save(str(initial / 'walk_model_vecnorm.pkl'))
            (initial / 'walk_params.json').write_text(json.dumps(
                {'walk_env_kwargs': walk_env_kwargs or {}, 'method': 'gait-example pretraining only'}))
        callback = CheckpointCallback(save_freq=max(checkpoint_steps // num_envs, 1),
            save_path=str(folder / 'checkpoints'), name_prefix=kind, save_vecnormalize=True)
        model.learn(total_timesteps=timesteps, progress_bar=progress_bar, callback=callback,
                    reset_num_timesteps=not resume)
        model.save(str(folder / f'{kind}_model'))
        env.save(str(folder / f'{kind}_model_vecnorm.pkl'))
        params = dict(kind=kind, reward_config=asdict(reward_config),
                      timesteps=timesteps, num_envs=num_envs,
                      ppo_kwargs=settings, seed=seed, oni_speed=oni_speed,
                      total_timesteps=model.num_timesteps, control_dt=CONTROL_DT,
                      elapsed_seconds=time.monotonic() - started,
                      environment_revision='walk-v2', walk_env_kwargs=walk_env_kwargs or {},
                      bootstrap_result=bootstrap_result, initial_log_std=initial_log_std,
                      freeze_normalization=freeze_normalization,
                      python=platform.python_version(),
                      packages={name: version(name) for name in
                          ['mujoco', 'gymnasium', 'stable-baselines3', 'torch', 'numpy']})
        (folder / f'{kind}_params.json').write_text(
            json.dumps(params, ensure_ascii=False, indent=2), encoding='utf-8')
    finally:
        env.close()
    print(f'保存完了: {folder / (kind + "_model.zip")}')


def evaluate_flee(save_dir, seeds=(100, 101, 102, 103, 104)):
    """Evaluate saved policies with frozen training statistics and fixed seeds."""
    folder = Path(save_dir)
    for name in ('walk_model.zip', 'walk_model_vecnorm.pkl', 'flee_model.zip',
                 'flee_model_vecnorm.pkl', 'flee_params.json'):
        if not (folder / name).is_file():
            raise FileNotFoundError(f'評価に必要な保存ファイルがありません: {folder / name}')
    params = json.loads((folder / 'flee_params.json').read_text(encoding='utf-8'))
    factory = lambda: Go2TagHierarchicalEnv(
        low_level_model_path=str(folder / 'walk_model'),
        low_level_vecnorm_path=str(folder / 'walk_model_vecnorm.pkl'),
        flee_config=FleeRewardConfig(**params['reward_config']),
        oni_speed=params['oni_speed'])
    base = make_vec_env(factory, n_envs=1)
    try:
        env = VecNormalize.load(str(folder / 'flee_model_vecnorm.pkl'), base)
        env.training = False
        env.norm_reward = False
        model = PPO.load(str(folder / 'flee_model'))
        results = []
        for seed in seeds:
            env.seed(seed)
            obs = env.reset()
            distances = []
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, dones, infos = env.step(action)
                info = infos[0]
                distances.append(info['oni_distance'])
                if dones[0]:
                    timeout = bool(info.get('TimeLimit.truncated', False))
                    tagged = bool(info['is_tagged'])
                    results.append(dict(seed=seed, survived_seconds=info['survived_seconds'],
                                        mean_distance=sum(distances) / len(distances),
                                        escaped=timeout, tagged=tagged,
                                        fell=not timeout and not tagged))
                    break
        return results
    finally:
        base.close()


def train_walk_curriculum(save_dir, reward_config=None, first_stage_steps=WALK_FORWARD_STEPS,
                          num_envs=4, ppo_kwargs=None, seed=0, final_refine_steps=200_000):
    """Reproduce the staged forward-walking experiment used for the local model.

    Stage 1: forward motion. Stage 2: forward motion with yaw commands.
    Stage 3: tighten straight-motion tracking with smaller PPO updates.
    This recipe is not an all-direction walking qualification.
    """
    from copy import deepcopy
    import shutil
    from roboquest.utils.reward_utils import WalkRewardConfig
    folder = Path(save_dir)
    reward = deepcopy(reward_config or WalkRewardConfig(**WALK_FORWARD_REWARD))
    settings = deepcopy(ppo_kwargs or WALK_FORWARD_PPO)
    if first_stage_steps < num_envs:
        raise ValueError('第1段階の学習ステップ数は環境数以上にしてください。')
    history = []
    print('1/3: 前進を学習します。')
    train_policy('walk', folder, reward, first_stage_steps, num_envs, settings, seed=seed,
        checkpoint_steps=first_stage_steps, walk_env_kwargs=deepcopy(WALK_FORWARD_ENV))
    # Evaluate and deploy the paired checkpoint at the requested step, rather
    # than silently substituting the end of the rounded-up PPO rollout.
    selected_step = (first_stage_steps // num_envs) * num_envs
    shutil.copy2(folder/'checkpoints'/f'walk_{selected_step}_steps.zip', folder/'walk_model.zip')
    shutil.copy2(folder/'checkpoints'/f'walk_vecnormalize_{selected_step}_steps.pkl',
                 folder/'walk_model_vecnorm.pkl')
    params = json.loads((folder/'walk_params.json').read_text())
    params['total_timesteps'] = selected_step
    params['selected_checkpoint'] = selected_step
    (folder/'walk_params.json').write_text(json.dumps(params, indent=2))
    history.append(dict(stage='forward', **params))
    print('2/3: 前進中の旋回指示も加えて学習します。')
    steering = deepcopy(WALK_FORWARD_ENV)
    steering['command_ranges']['omega'] = [-.5, .5]
    train_policy('walk', folder, reward, 500_000, num_envs, settings, seed=seed, resume=True,
                 walk_env_kwargs=steering)
    history.append(dict(stage='steering', **json.loads((folder/'walk_params.json').read_text())))
    print('3/3: 前進速度と直進時の角速度を調整します。')
    reward.linear_tracking_variance = .09
    reward.angular_tracking_variance = .04
    reward.ang_vel_weight *= 2
    straight = deepcopy(WALK_FORWARD_ENV)
    straight['command_ranges']['vx'] = [.4, .4]
    train_policy('walk', folder, reward, 50_000, num_envs, settings, seed=seed, resume=True,
        walk_env_kwargs=straight, allow_reward_change=True, initial_log_std=-1.8,
        ppo_overrides={'learning_rate': settings['learning_rate']/10,
                       'target_kl': .003, 'n_epochs': 5})
    history.append(dict(stage='straight', **json.loads((folder/'walk_params.json').read_text())))
    if final_refine_steps:
        print('仕上げ: 直進の追加学習をします。')
        train_policy('walk', folder, reward, final_refine_steps, num_envs, settings,
                     seed=seed, resume=True, checkpoint_steps=50_000)
        history.append(dict(stage='straight_continued', **json.loads((folder/'walk_params.json').read_text())))
    (folder/'walk_curriculum.json').write_text(json.dumps(history, indent=2, ensure_ascii=False))
