"""Check stop / forward / stop / forward without resetting the robot."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from roboquest.envs.go2_walk_env import Go2WalkEnv, CONTROL_DT


def evaluate_transitions(folder, seeds=(200, 203, 209), seconds=20):
    folder = Path(folder)
    torch.set_num_threads(1)
    params = json.loads((folder/'walk_params.json').read_text())
    raw = Go2WalkEnv(randomize_cmd=False, **params['walk_env_kwargs'])
    norm = VecNormalize.load(str(folder/'walk_model_vecnorm.pkl'), DummyVecEnv([lambda: raw]))
    norm.training = False
    model = PPO.load(folder/'walk_model', device='cpu')
    rows = []
    try:
        for seed in seeds:
            raw.set_vel_cmd(0, 0, 0)
            raw.reset(seed=seed)
            for segment, vx in enumerate((0., .4, 0., .4)):
                raw.set_vel_cmd(vx, 0., 0.)
                obs = raw._get_obs().astype(np.float32)
                velocities, contacts, knees = [], [], []
                fell = False
                for step in range(round(seconds/CONTROL_DT)):
                    action, _ = model.predict(norm.normalize_obs(obs), deterministic=True)
                    obs, _, fell, _, _ = raw.step(action)
                    posture = raw.posture_metrics()
                    contacts.append(posture['nonfoot_contacts'] > 0)
                    knees.append(float(np.min(posture['knee_heights'][2:])))
                    if step >= 50:
                        v = raw._world_to_body(raw.data.qvel[:3])
                        velocities.append([v[0], v[1], raw.data.qvel[5]])
                    if fell:
                        break
                mean = np.mean(velocities, axis=0) if velocities else np.full(3, np.nan)
                valid = bool(not fell and not any(contacts) and np.all(np.isfinite(mean))
                    and np.all(np.abs(mean-[vx,0,0]) <= [max(.06, .35*vx), .06, .12]))
                rows.append(dict(seed=seed, segment=segment, target_vx=vx,
                    mean_velocity=mean.tolist() if np.all(np.isfinite(mean)) else None,
                    fell=bool(fell), nonfoot_contact_fraction=float(np.mean(contacts)),
                    min_hind_knee_height_m=min(knees), passed=valid))
                if fell:
                    break
        report = dict(rows=rows, passed=len(rows)==len(seeds)*4 and all(r['passed'] for r in rows),
                      segment_seconds=seconds, warmup_seconds=1,
                      contact_measurement='all control steps, including command transitions')
        (folder/'walk_transition_evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps(report, indent=2))
        return report
    finally:
        norm.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder')
    args = parser.parse_args()
    evaluate_transitions(args.folder)
