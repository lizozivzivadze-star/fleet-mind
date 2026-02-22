"""
Fleet Mind — PPO Training Script
=================================
Usage:
    python train/train_agents.py --mode both --timesteps 300000

Requirements:
    pip install stable-baselines3 gymnasium numpy torch

Outputs:
    models/dual_usv_ppo.zip      — trained model
    logs/training_log.json       — live stats (read by backend)
    logs/latest_replay.json      — evaluation replay frames
"""

import os, sys, json, argparse, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.dual_usv_env import DualUSVEnv

# Try importing SB3 — give clear error if missing
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("[WARNING] stable-baselines3 not installed.")
    print("          Run:  pip install stable-baselines3")


# ────────────────────────────────────────────────
# CALLBACK — writes live stats for the dashboard
# ────────────────────────────────────────────────
class LiveStatsCallback(BaseCallback):
    def __init__(self, log_path="logs/training_log.json", verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.ep_rewards   = []
        self.ep_coverages = []
        self.ep_threats   = []
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [False])
        infos = self.locals.get("infos", [{}])
        rewards = self.locals.get("rewards", [0])

        for done, info, r in zip(dones, infos, rewards):
            if done:
                self.ep_rewards.append(float(r))
                self.ep_coverages.append(info.get("coverage_pct", 0))
                self.ep_threats.append(info.get("threats_neutralized", 0))

                n = max(1, min(30, len(self.ep_rewards)))
                log = {
                    "episode":       len(self.ep_rewards),
                    "mean_reward":   round(float(np.mean(self.ep_rewards[-n:])), 3),
                    "mean_coverage": round(float(np.mean(self.ep_coverages[-n:])), 2),
                    "mean_threats":  round(float(np.mean(self.ep_threats[-n:])), 2),
                    "timesteps":     int(self.num_timesteps),
                    "updated_at":    time.strftime("%H:%M:%S"),
                }
                with open(self.log_path, "w") as f:
                    json.dump(log, f, indent=2)

                if self.verbose:
                    print(f"  Ep {log['episode']:4d} | "
                          f"cov={log['mean_coverage']:5.1f}% | "
                          f"reward={log['mean_reward']:7.3f} | "
                          f"threats={log['mean_threats']:.1f}")
        return True


# ────────────────────────────────────────────────
# TRAIN
# ────────────────────────────────────────────────
def train(timesteps: int = 300_000,
          save_path: str = "models/dual_usv_ppo",
          verbose: int = 1):

    if not SB3_AVAILABLE:
        print("Cannot train — stable-baselines3 is not installed.")
        return None

    print(f"\n{'='*55}")
    print("  FLEET MIND — RL TRAINING")
    print(f"  timesteps : {timesteps:,}")
    print(f"  save path : {save_path}.zip")
    print(f"{'='*55}\n")

    env = Monitor(DualUSVEnv())
    check_env(env, warn=True)

    model = PPO(
        policy          = "MlpPolicy",
        env             = env,
        learning_rate   = 3e-4,
        n_steps         = 2048,
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.01,     # encourages exploration
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        verbose         = verbose,
        tensorboard_log = "logs/tensorboard/",
        policy_kwargs   = dict(net_arch=[256, 256]),   # two hidden layers
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    callback = LiveStatsCallback(verbose=verbose)

    model.learn(total_timesteps=timesteps, callback=callback)
    model.save(save_path)

    print(f"\n✓ Model saved → {save_path}.zip")
    return model


# ────────────────────────────────────────────────
# EVALUATE + SAVE REPLAY
# ────────────────────────────────────────────────
def evaluate(model_path: str = "models/dual_usv_ppo",
             episodes:   int = 5,
             replay_path: str = "logs/latest_replay.json"):

    if not SB3_AVAILABLE:
        print("Cannot evaluate — stable-baselines3 is not installed.")
        return

    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found at {model_path}.zip — run training first.")
        return

    print(f"\nEvaluating {episodes} episodes …")
    env   = DualUSVEnv()
    model = PPO.load(model_path)

    all_replays = []
    all_scores  = []

    for ep in range(episodes):
        obs, _ = env.reset()
        frames = []
        done   = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
            frames.append(env.render())

        all_replays.append(frames)
        all_scores.append(info)

        print(f"  Episode {ep+1}: coverage={info['coverage_pct']:.1f}%  "
              f"threats={info['threats_neutralized']}/3  "
              f"reward={info['total_reward']:.2f}")

    os.makedirs(os.path.dirname(replay_path), exist_ok=True)
    with open(replay_path, "w") as f:
        json.dump({"replays": all_replays, "scores": all_scores}, f)

    print(f"\n✓ Replay saved → {replay_path}")
    return all_replays, all_scores


# ────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fleet Mind RL Trainer")
    parser.add_argument("--mode",       choices=["train", "eval", "both"], default="both")
    parser.add_argument("--timesteps",  type=int,  default=300_000)
    parser.add_argument("--episodes",   type=int,  default=5)
    parser.add_argument("--model",      type=str,  default="models/dual_usv_ppo")
    parser.add_argument("--verbose",    type=int,  default=1)
    args = parser.parse_args()

    if args.mode in ["train", "both"]:
        train(timesteps=args.timesteps,
              save_path=args.model,
              verbose=args.verbose)

    if args.mode in ["eval", "both"]:
        evaluate(model_path=args.model,
                 episodes=args.episodes)
