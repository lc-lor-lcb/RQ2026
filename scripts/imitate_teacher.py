"""Train a small MLP to imitate teacher actions.

This produces a PyTorch checkpoint for analysis. Integrating it into SB3 PPO as
an initialization step is intentionally left explicit so the submission pipeline
does not silently depend on a non-standard policy format.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TeacherPolicy(nn.Module):
    def __init__(self, obs_dim: int = 10, act_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, act_dim),
            nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="runs/tier2_teacher/teacher_data.npz")
    parser.add_argument("--output", default="runs/tier2_teacher/teacher_policy.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    data = np.load(args.data)
    obs = torch.tensor(data["obs"], dtype=torch.float32)
    actions = torch.tensor(data["actions"], dtype=torch.float32)
    loader = DataLoader(TensorDataset(obs, actions), batch_size=args.batch_size, shuffle=True)
    model = TeacherPolicy(obs.shape[1], actions.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()
    for epoch in range(args.epochs):
        total = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
        print(f"epoch {epoch + 1}/{args.epochs} loss={total / len(obs):.6f}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "obs_dim": obs.shape[1], "act_dim": actions.shape[1]}, out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
