import argparse
import json
import os
import time

import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from aura.gym_env import JointGymEnv
from aura.traces import load_all

torch.set_num_threads(1)


def build_model(algo, venv, seed, gamma, ent_coef):
    common = dict(seed=seed, verbose=0, gamma=gamma, policy_kwargs=dict(net_arch=[128, 128]))
    if algo == "PPO":
        return PPO("MlpPolicy", venv, n_steps=2048, batch_size=256, gae_lambda=0.95,
                   ent_coef=ent_coef, learning_rate=3e-4, **common)
    return DQN("MlpPolicy", venv, buffer_size=100000, batch_size=128, learning_rate=5e-4,
               exploration_fraction=0.4, exploration_final_eps=0.03, learning_starts=2000,
               target_update_interval=1000, **common)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=os.environ.get("AURA_DATA", "data"))
    p.add_argument("--sources", required=True, help="comma-separated training datasets")
    p.add_argument("--algo", default="PPO", choices=["PPO", "DQN"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=200000)
    p.add_argument("--no-context", action="store_true")
    p.add_argument("--migration-delay", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--ent-coef", type=float, default=0.003)
    p.add_argument("--mode", default="factored",
                   choices=["flat", "factored", "residual", "residual_ctx", "experts"],
                   help="action space; residual/experts are learning-augmented (see aura/advisors.py)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    sources = args.sources.split(",")
    train, _ = load_all(args.data_root, sources)
    context = not args.no_context
    mode = "flat" if args.algo == "DQN" else args.mode
    venv = DummyVecEnv([lambda: JointGymEnv(train, context=context,
                                            migration_delay=args.migration_delay,
                                            seed=args.seed, mode=mode)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    start = time.time()
    model = build_model(args.algo, venv, args.seed, args.gamma, args.ent_coef)
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    os.makedirs(args.out, exist_ok=True)
    model.save(os.path.join(args.out, "model"))
    venv.save(os.path.join(args.out, "vecnormalize.pkl"))
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({**vars(args), "context": context, "mode": mode, "train_seconds": time.time() - start,
                   "train_windows": {k: len(v) for k, v in train.items()}}, f, indent=2)


if __name__ == "__main__":
    main()
