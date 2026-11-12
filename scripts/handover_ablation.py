import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from aura.baselines import SAFETY, _context_edge, _highest_below, context_joint
from aura.evaluation import evaluate, paired_bootstrap
from aura.traces import load_dataset


def context_joint_no_handover(env, obs=None):
    edge = _context_edge(env)
    rate = SAFETY * min(env.radio_estimate(), env.edge_available()[edge])
    if env.buffer < 8.0:
        rate *= 0.7
    return env.encode(_highest_below(rate), edge)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=os.environ.get("AURA_DATA", "data"))
    p.add_argument("--out", default="results/handover_ablation.csv")
    args = p.parse_args()
    _, test = load_dataset("5g", args.data_root)
    with_ho = pd.DataFrame(evaluate(context_joint, test)).qoe.values
    without = pd.DataFrame(evaluate(context_joint_no_handover, test)).qoe.values
    stats = paired_bootstrap(with_ho, without)
    stats["dataset"] = "5g"
    pd.DataFrame([stats]).to_csv(args.out, index=False)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()})


if __name__ == "__main__":
    main()
