"""
Fleet Mind — Dual USV Patrol Environment
Compatible with Gymnasium + Stable-Baselines3 (PPO)

Install deps:
    pip install stable-baselines3 gymnasium numpy
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DualUSVEnv(gym.Env):
    """
    Two USVs coordinate to:
      1. Cover a 10×10 patrol grid
      2. Maintain formation (50–150 unit spacing)
      3. Neutralize up to 3 threats

    Observation (116 floats, all normalized 0-1):
        usv1_x, usv1_y, usv1_heading           (3)
        usv2_x, usv2_y, usv2_heading           (3)
        formation_distance                      (1)
        threat_1_x, threat_1_y, threat_1_active (3)
        threat_2_x, threat_2_y, threat_2_active (3)
        threat_3_x, threat_3_y, threat_3_active (3)
        coverage_grid (10×10 flattened)        (100)
        Total: 116

    Action (MultiDiscrete [4, 4]):
        Per USV: 0=forward, 1=turn_left, 2=turn_right, 3=hold
    """

    metadata = {"render_modes": ["state_dict"]}

    WORLD       = 500
    CELL_SIZE   = 50
    GRID_CELLS  = 10
    MAX_STEPS   = 500
    THREAT_N    = 3
    MIN_FORM    = 50
    MAX_FORM    = 150
    THREAT_RANGE = 42
    SCAN_RADIUS = 55
    USV_SPEED   = 10
    TURN_RATE   = 18   # degrees per step
    COLLIDE_DIST = 22

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        obs_dim = 3 + 3 + 1 + self.THREAT_N * 3 + self.GRID_CELLS ** 2
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([4, 4])

        self.np_random = np.random.default_rng()
        self._init_state()

    # ------------------------------------------------------------------
    def _init_state(self):
        self.usv1_pos     = np.array([60.0, 150.0])
        self.usv2_pos     = np.array([60.0, 350.0])
        self.usv1_heading = 0.0
        self.usv2_heading = 0.0

        # Spread threats across three columns so each episode differs
        self.threats = []
        xs = np.random.uniform(200, 460, self.THREAT_N)
        ys = np.random.uniform(30, 470, self.THREAT_N)
        for i in range(self.THREAT_N):
            self.threats.append({
                "pos": np.array([xs[i], ys[i]]),
                "active": True
            })

        self.coverage = np.zeros(
            (self.GRID_CELLS, self.GRID_CELLS), dtype=np.float32
        )
        self.step_count   = 0
        self.total_reward = 0.0

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self._init_state()
        return self._obs(), {}

    # ------------------------------------------------------------------
    def _obs(self):
        obs = []
        obs += (self.usv1_pos / self.WORLD).tolist()
        obs += [self.usv1_heading / 360.0]
        obs += (self.usv2_pos / self.WORLD).tolist()
        obs += [self.usv2_heading / 360.0]

        dist = float(np.linalg.norm(self.usv1_pos - self.usv2_pos))
        obs += [min(dist / self.WORLD, 1.0)]

        for t in self.threats:
            obs += (t["pos"] / self.WORLD).tolist()
            obs += [1.0 if t["active"] else 0.0]

        obs += self.coverage.flatten().tolist()
        return np.array(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    def _move(self, pos, heading, action):
        if action == 0:   # forward
            rad = np.radians(heading)
            pos = pos + np.array([np.cos(rad), np.sin(rad)]) * self.USV_SPEED
            pos = np.clip(pos, 8, self.WORLD - 8)
        elif action == 1: # left
            heading = (heading - self.TURN_RATE) % 360
        elif action == 2: # right
            heading = (heading + self.TURN_RATE) % 360
        # 3 = hold
        return pos, heading

    # ------------------------------------------------------------------
    def _update_coverage(self, pos):
        reward = 0.0
        cx = int(np.clip(pos[0] / self.CELL_SIZE, 0, self.GRID_CELLS - 1))
        cy = int(np.clip(pos[1] / self.CELL_SIZE, 0, self.GRID_CELLS - 1))

        # Direct cell
        if self.coverage[cy, cx] < 1.0:
            was_zero = self.coverage[cy, cx] == 0.0
            self.coverage[cy, cx] = 1.0
            if was_zero:
                reward += 0.3

        # Scan radius — partially covers neighbors
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.GRID_CELLS and 0 <= ny < self.GRID_CELLS:
                    cell_cx = nx * self.CELL_SIZE + self.CELL_SIZE / 2
                    cell_cy = ny * self.CELL_SIZE + self.CELL_SIZE / 2
                    if np.linalg.norm(pos - np.array([cell_cx, cell_cy])) < self.SCAN_RADIUS:
                        if self.coverage[ny, nx] < 0.5:
                            self.coverage[ny, nx] = 0.5
        return reward

    # ------------------------------------------------------------------
    def step(self, action):
        self.usv1_pos, self.usv1_heading = self._move(
            self.usv1_pos, self.usv1_heading, int(action[0])
        )
        self.usv2_pos, self.usv2_heading = self._move(
            self.usv2_pos, self.usv2_heading, int(action[1])
        )

        reward = 0.0

        # Coverage
        reward += self._update_coverage(self.usv1_pos)
        reward += self._update_coverage(self.usv2_pos)

        # Formation
        dist = float(np.linalg.norm(self.usv1_pos - self.usv2_pos))
        if dist < self.COLLIDE_DIST:
            reward -= 0.2
        elif self.MIN_FORM <= dist <= self.MAX_FORM:
            reward += 0.2
        elif dist > self.MAX_FORM:
            reward -= 0.05

        # Threats
        for t in self.threats:
            if not t["active"]:
                continue
            d1 = float(np.linalg.norm(self.usv1_pos - t["pos"]))
            d2 = float(np.linalg.norm(self.usv2_pos - t["pos"]))
            if d1 < self.THREAT_RANGE or d2 < self.THREAT_RANGE:
                t["active"] = False
                reward += 1.0

        self.step_count   += 1
        self.total_reward += reward

        terminated = self.step_count >= self.MAX_STEPS
        truncated  = False

        coverage_pct = float(
            np.sum(self.coverage > 0) / (self.GRID_CELLS ** 2) * 100
        )
        threats_done = sum(1 for t in self.threats if not t["active"])

        info = {
            "step":                self.step_count,
            "coverage_pct":        round(coverage_pct, 2),
            "threats_neutralized": threats_done,
            "formation_dist":      round(dist, 1),
            "total_reward":        round(self.total_reward, 3),
        }

        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        """Returns JSON-serializable state dict for the frontend."""
        return {
            "usv1": {
                "x": round(float(self.usv1_pos[0]), 1),
                "y": round(float(self.usv1_pos[1]), 1),
                "heading": round(self.usv1_heading, 1),
            },
            "usv2": {
                "x": round(float(self.usv2_pos[0]), 1),
                "y": round(float(self.usv2_pos[1]), 1),
                "heading": round(self.usv2_heading, 1),
            },
            "threats": [
                {
                    "x": round(float(t["pos"][0]), 1),
                    "y": round(float(t["pos"][1]), 1),
                    "active": t["active"],
                }
                for t in self.threats
            ],
            "coverage": self.coverage.tolist(),
            "step": self.step_count,
        }
