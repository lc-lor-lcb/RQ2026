"""Optional gait-example pretraining, followed by PPO (not a learned policy itself)."""
import numpy as np


def gait_example(obs):
    """A periodic teaching action inferred from the preceding action, without a hidden clock."""
    obs = np.asarray(obs)
    vx, vy, yaw = obs[:3]
    if np.linalg.norm(obs[:3]) < .05:
        return np.zeros(12, dtype=np.float32)
    previous = obs[-12:].reshape(4, 3)
    diagonal = np.array([1., -1., -1., 1.])
    side = np.array([1., -1., 1., -1.])
    amp_x = np.clip((vx + side * .15 * yaw) / .3, -1., 1.)
    amp_y = np.clip(-vy / .3, -1., 1.)
    cos_phase = np.mean(previous[:, 2] * diagonal)
    estimates = []
    for leg in range(4):
        if abs(amp_x[leg]) > .1 and abs(previous[leg, 1]) < .99:
            estimates.append((previous[leg, 1] + .5 * previous[leg, 2] - .3)
                             / amp_x[leg] * diagonal[leg])
        if abs(amp_y) > .1:
            estimates.append(previous[leg, 0] / amp_y * diagonal[leg])
    sin_phase = np.mean(estimates) if estimates else 0.
    phase = np.arctan2(sin_phase, cos_phase) if np.linalg.norm(previous) > .01 else 0.
    phase += 2 * np.pi * 1.5 * .02
    sine, cosine = np.sin(phase) * diagonal, np.cos(phase) * diagonal
    action = np.zeros((4, 3))
    action[:, 0] = amp_y * sine
    action[:, 1] = amp_x * sine - .5 * cosine + .3
    action[:, 2] = cosine
    return np.clip(action.reshape(-1), -1, 1).astype(np.float32)


def pretrain_examples(model, env, seed=0, sample_steps=20000, updates=3000):
    """Fit the policy network to gait examples; PPO still performs subsequent training."""
    import torch
    from roboquest.envs.go2_walk_env import Go2WalkEnv
    from roboquest.utils.reward_utils import WalkRewardConfig
    rng = np.random.default_rng(seed)
    teacher_env = Go2WalkEnv(randomize_cmd=False, action_scale=.6,
        reward_config=WalkRewardConfig(feet_gait_weight=0, foot_slip_weight=0))
    observations, targets = [], []
    obs = None
    for step in range(sample_steps):
        if step % 500 == 0 or obs is None:
            teacher_env.set_vel_cmd(float(rng.uniform(.25, .5)), 0, 0)
            obs, _ = teacher_env.reset(seed=seed + step)
        observations.append(obs.copy())
        action = gait_example(obs)
        targets.append(action)
        obs, _, terminated, _, _ = teacher_env.step(action)
        if terminated:
            obs = None
    teacher_env.close()
    observations = np.asarray(observations, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    env.obs_rms.update(observations)
    # The teacher depends on command and preceding action. Add off-trajectory
    # examples to make the learned oscillator recover from small policy errors.
    augmented = observations.copy()
    augmented[:, 3:33] += rng.normal(0, .1, augmented[:, 3:33].shape)
    augmented[:, -12:] += rng.normal(0, .04, augmented[:, -12:].shape)
    augmented_targets = np.stack([gait_example(row) for row in augmented])
    observations = np.concatenate([observations, augmented])
    targets = np.concatenate([targets, augmented_targets])
    x = torch.as_tensor(env.normalize_obs(observations), device=model.device)
    y = torch.as_tensor(targets, device=model.device)
    params = list(model.policy.mlp_extractor.policy_net.parameters()) + list(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    for update in range(updates):
        idx = torch.as_tensor(rng.integers(0, len(x), 512), device=model.device)
        predicted = model.policy._predict(x[idx], deterministic=True)
        loss = (predicted - y[idx]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (update + 1) % 500 == 0:
            print(f'Gait example update {update+1}: MSE={loss.item():.6f}', flush=True)
    with torch.no_grad():
        model.policy.log_std.fill_(-2.)
    return dict(sample_steps=sample_steps, updates=updates, mse=float(loss.item()))
