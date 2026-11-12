import argparse
import collections
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import evaluate as ev
from aura.advisors import MODES, RESIDUAL_STEPS
from aura.evaluation import EVAL_SEED, run_episode
from aura.traces import DATASETS, load_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=os.environ.get("AURA_DATA", "data"))
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--dataset", default="5g")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    train, test = load_all(args.data_root, DATASETS)
    rows = []
    original = ev.to_env_action
    for run_dir in args.runs:
        policy, cfg, mode = ev.load_agent(run_dir, train)
        if mode not in ("residual", "residual_ctx"):
            raise SystemExit(f"{run_dir}: action statistics need a residual agent")
        planner, planner_dim = MODES[mode][:2]
        shifts, edges = collections.Counter(), collections.Counter()

        def recording(env, action, mode_, _orig=original):
            shifts[int(RESIDUAL_STEPS[int(action[0])])] += 1
            edges[int(action[1])] += 1
            return _orig(env, action, mode_)

        ev.to_env_action = recording
        for i, w in enumerate(test[args.dataset]):
            run_episode(policy, w, cfg["context"], cfg["migration_delay"], seed=EVAL_SEED + i,
                        planner=planner, planner_dim=planner_dim)
        ev.to_env_action = original
        n = sum(shifts.values())
        row = {"run": os.path.basename(run_dir.rstrip("/")), "dataset": args.dataset, "chunks": n}
        row.update({f"shift_{k:+d}": shifts.get(k, 0) / n for k in sorted(set(RESIDUAL_STEPS))})
        row.update({f"edge_{k}": edges.get(k, 0) / n for k in range(3)})
        rows.append(row)
        print(row)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
