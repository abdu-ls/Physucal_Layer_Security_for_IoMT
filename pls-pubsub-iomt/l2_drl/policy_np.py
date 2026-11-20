"""NumPy inference for an exported Dueling DQN (policy.npz). No PyTorch needed.
Used by evaluate.py and by the L3 testbed orchestrator."""
from __future__ import annotations

import numpy as np


class NumpyDuelingPolicy:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.hidden = [(z[f"h{i}_w"], z[f"h{i}_b"]) for i in range(int(z["n_hidden"]))]
        self.v = (z["v_w"], z["v_b"])
        self.a = (z["a_w"], z["a_b"])

    def q(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        for w, b in self.hidden:
            x = np.maximum(x @ w.T + b, 0.0)
        v = x @ self.v[0].T + self.v[1]
        adv = x @ self.a[0].T + self.a[1]
        return v + adv - adv.mean(axis=-1, keepdims=True)

    def __call__(self, env, obs, mask):
        q = self.q(obs)
        q = np.where(mask, q, -np.inf)
        return int(np.argmax(q))

    def act(self, obs, mask):
        q = np.where(mask, self.q(obs), -np.inf)
        return int(np.argmax(q))
