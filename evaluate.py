import argparse
import json
import os

import pandas as pd
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from aura.advisors import MODES, to_env_action
from aura.baselines import HEURISTICS
from aura.evaluation import evaluate
from aura.gym_env import JointGymEnv
from aura.traces import DATASETS, load_all

torch.set_num_threads(1)


def load_agent(run_dir, train_windows):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    algo = PPO if cfg["algo"] == "PPO" else DQN
    model = algo.load(os.path.join(run_dir, "model"), device="cpu")
    mode = cfg.get("mode") or ("residual" if cfg.get("residual") else
                                ("factored" if cfg.get("factored") else "flat"))
    dummy = DummyVecEnv([lambda: JointGymEnv(train_windows, context=cfg["context"], mode=mode)])
    venv = VecNormalize.load(os.path.join(run_dir, "vecnormalize.pkl"), dummy)
    venv.training = False
    venv.norm_reward = False

    def policy(env, obs):
        action, _ = model.predict(venv.normalize_obs(obs), deterministic=True)
        return to_env_action(env, action, mode)

    return policy, cfg, mode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=os.environ.get("AURA_DATA", "data"))
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--baselines", action="store_true")
    p.add_argument("--migration-delay", type=float, default=1.0)
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--out", required=True)
    args = p.parse_args()

    datasets = args.datasets.split(",")
    train, test = load_all(args.data_root, datasets)
    rows = []

    if args.baselines:
        for name, fn in HEURISTICS.items():
            for d in datasets:
                for r in evaluate(fn, test[d], context=True, migration_delay=args.migration_delay):
                    rows.append({**r, "policy": name, "seed": -1, "run": name,
                                 "migration_delay": args.migration_delay})

    for run_dir in args.runs:
        policy, cfg, mode = load_agent(run_dir, train)
        planner, planner_dim = MODES[mode][:2] if mode in MODES else (None, 0)
        label = os.path.basename(run_dir.rstrip("/"))
        for d in datasets:
            for r in evaluate(policy, test[d], context=cfg["context"],
                              migration_delay=cfg["migration_delay"],
                              planner=planner, planner_dim=planner_dim):
                name = cfg["algo"] + ("" if mode in ("flat", "factored") else f"-{mode}")
                rows.append({**r, "policy": name, "seed": cfg["seed"], "run": label, "mode": mode,
                             "sources": cfg["sources"], "context": cfg["context"],
                             "migration_delay": cfg["migration_delay"]})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
