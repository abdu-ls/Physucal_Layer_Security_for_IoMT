"""Dueling Double DQN with action masking (PyTorch)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=(128, 128)):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            d = h
        self.hidden = nn.ModuleList(layers)
        self.v = nn.Linear(d, 1)
        self.a = nn.Linear(d, n_actions)

    def forward(self, x):
        for l in self.hidden:
            x = F.relu(l(x))
        adv = self.a(x)
        return self.v(x) + adv - adv.mean(dim=-1, keepdim=True)


class ReplayBuffer:
    def __init__(self, cap, obs_dim, n_actions):
        self.cap, self.i, self.full = cap, 0, False
        self.o = np.zeros((cap, obs_dim), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.o2 = np.zeros((cap, obs_dim), np.float32)
        self.m2 = np.zeros((cap, n_actions), bool)
        self.d = np.zeros(cap, np.float32)

    def add(self, o, a, r, o2, m2, d):
        k = self.i
        self.o[k], self.a[k], self.r[k], self.o2[k], self.m2[k], self.d[k] = o, a, r, o2, m2, d
        self.i = (k + 1) % self.cap
        self.full |= self.i == 0

    def __len__(self):
        return self.cap if self.full else self.i

    def sample(self, n, rng):
        idx = rng.integers(0, len(self), size=n)
        return self.o[idx], self.a[idx], self.r[idx], self.o2[idx], self.m2[idx], self.d[idx]


class DuelingDQNAgent:
    def __init__(self, obs_dim, n_actions, cfg, seed=0, device=None):
        self.cfg = cfg
        self.n_actions = n_actions
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.q = DuelingQNet(obs_dim, n_actions, cfg["hidden"]).to(self.device)
        self.tgt = DuelingQNet(obs_dim, n_actions, cfg["hidden"]).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg["lr"])
        self.buf = ReplayBuffer(cfg["buffer_size"], obs_dim, n_actions)
        self.steps = 0

    def epsilon(self):
        c = self.cfg
        f = min(1.0, self.steps / c["eps_decay_steps"])
        return c["eps_start"] + f * (c["eps_end"] - c["eps_start"])

    def act(self, obs, mask, greedy=False):
        if not greedy and self.rng.random() < self.epsilon():
            return int(self.rng.choice(np.flatnonzero(mask)))
        with torch.no_grad():
            q = self.q(torch.as_tensor(obs, device=self.device).unsqueeze(0)).squeeze(0).cpu().numpy()
        q[~mask] = -np.inf
        return int(np.argmax(q))

    def update(self):
        c = self.cfg
        self.steps += 1
        if len(self.buf) < c["warmup_steps"]:
            return None
        o, a, r, o2, m2, d = self.buf.sample(c["batch_size"], self.rng)
        dev = self.device
        o, o2 = torch.as_tensor(o, device=dev), torch.as_tensor(o2, device=dev)
        a, r, d = torch.as_tensor(a, device=dev), torch.as_tensor(r, device=dev), torch.as_tensor(d, device=dev)
        m2 = torch.as_tensor(m2, device=dev)
        q = self.q(o).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next_online = self.q(o2).masked_fill(~m2, -1e9)
            a2 = q_next_online.argmax(dim=1, keepdim=True)            # double DQN
            q_next = self.tgt(o2).gather(1, a2).squeeze(1)
            y = r + c["gamma"] * (1 - d) * q_next
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        if self.steps % c["target_update_steps"] == 0:
            self.tgt.load_state_dict(self.q.state_dict())
        return float(loss.item())

    def save(self, path):
        torch.save(self.q.state_dict(), path)

    def export_npz(self, path):
        """Weights for NumPy inference (l2_drl/policy_np.py and the L3 orchestrator)."""
        sd = {k: v.detach().cpu().numpy() for k, v in self.q.state_dict().items()}
        n_hidden = len(self.q.hidden)
        out = {"n_hidden": np.array(n_hidden)}
        for i in range(n_hidden):
            out[f"h{i}_w"], out[f"h{i}_b"] = sd[f"hidden.{i}.weight"], sd[f"hidden.{i}.bias"]
        out["v_w"], out["v_b"] = sd["v.weight"], sd["v.bias"]
        out["a_w"], out["a_b"] = sd["a.weight"], sd["a.bias"]
        np.savez(path, **out)
