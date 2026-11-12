import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from aura.evaluation import paired_bootstrap
from aura.traces import DATASETS

METRICS = ["qoe", "bitrate", "rebuffer_ratio", "latency_ms", "migrations_per_100",
           "bitrate_switches_per_100"]
GROUP_NAMES = {"all": "Learned (pooled)", "all_pure": "From scratch", "all_noctx": "Learned, no context",
               "lodo": "Learned (unseen dataset)"}


def load(results_dir):
    frames = [pd.read_csv(f) for f in glob.glob(os.path.join(results_dir, "*.csv"))
              if os.path.basename(f).startswith(("baselines_", "agent_"))]
    return pd.concat(frames, ignore_index=True)


def group_of(run):
    if run.startswith("lodo_"):
        return "lodo"
    if run.startswith("single_"):
        return "single_" + run.split("_")[1]
    return run.rsplit("_s", 1)[0]


def agent_vs_baselines(agent, base, dataset):
    a = agent[agent.dataset == dataset]
    per_window = a.groupby("window").qoe.mean().sort_index()
    seed_means = a.groupby("seed").qoe.mean()
    b = base[base.dataset == dataset]
    heur = b.groupby("policy").qoe.mean()
    best = heur.idxmax()
    best_w = b[b.policy == best].set_index("window").qoe.sort_index()
    stats = paired_bootstrap(per_window.values, best_w.loc[per_window.index].values)
    mpc_w = b[b.policy == "MPC-Joint"].set_index("window").qoe.sort_index()
    vs_mpc = paired_bootstrap(per_window.values, mpc_w.loc[per_window.index].values)
    row = {"dataset": dataset, "agent_qoe": per_window.mean(),
           "agent_seed_std": seed_means.std(ddof=1) if len(seed_means) > 1 else 0.0,
           "seeds_beating_best": int((seed_means > heur.max()).sum()),
           "n_seeds": len(seed_means), "best_heuristic": best, "best_heuristic_qoe": heur.max(),
           "gain_vs_mpc": vs_mpc["mean_diff"], "p_vs_mpc": vs_mpc["p_value"],
           **stats}
    for m in METRICS[1:]:
        row["agent_" + m] = a[m].mean()
        row["best_" + m] = b[b.policy == best][m].mean()
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    args = p.parse_args()

    df = load(args.results)
    df["group"] = df.run.map(group_of)
    base_all = df[df.seed == -1]
    agents = df[df.seed >= 0]
    summary = {}

    def baselines_at(delay):
        return base_all[np.isclose(base_all.migration_delay, delay)]

    base = baselines_at(1.0)
    summary["baselines"] = base.pivot_table(index="policy", columns="dataset", values="qoe").round(3).to_dict()

    rows = []
    for group in ["all", "all_noctx", "all_pure", "lodo"]:
        g = agents[agents.group == group]
        if g.empty:
            continue
        for d in DATASETS:
            sub = g[g.run.str.startswith(f"lodo_{d}_")] if group == "lodo" else g
            if sub[sub.dataset == d].empty:
                continue
            rows.append({"group": group, **agent_vs_baselines(sub, base, d)})
    main_table = pd.DataFrame(rows)
    main_table.to_csv(os.path.join(args.results, "table_main.csv"), index=False)
    summary["main"] = main_table.round(4).to_dict(orient="records")

    matrix = {}
    for src in DATASETS:
        g = agents[agents.group == f"single_{src}"]
        if g.empty:
            continue
        matrix[src] = {d: agent_vs_baselines(g, base, d)["mean_diff"] for d in DATASETS}
    if matrix:
        mat = pd.DataFrame(matrix).T
        mat.to_csv(os.path.join(args.results, "table_transfer_matrix.csv"))
        summary["transfer_matrix"] = mat.round(4).to_dict()

    delay_rows = []
    for delay in sorted(agents.migration_delay.unique()):
        g = agents[(agents.group.isin(["all", f"all_delay{delay}"])) & np.isclose(agents.migration_delay, delay)]
        b = baselines_at(delay)
        if g.empty or b.empty:
            continue
        for d in DATASETS:
            r = agent_vs_baselines(g, b, d)
            ctx = b[(b.policy == "Ctx-Joint") & (b.dataset == d)]
            delay_rows.append({"migration_delay": delay, **r,
                               "ctx_joint_qoe": ctx.qoe.mean(),
                               "ctx_joint_migrations": ctx.migrations_per_100.mean()})
    if delay_rows:
        dt = pd.DataFrame(delay_rows)
        dt.to_csv(os.path.join(args.results, "table_delay.csv"), index=False)
        summary["delay"] = dt.round(4).to_dict(orient="records")

    per_policy = base.groupby(["policy", "dataset"]).qoe.mean().unstack()
    for group in ["all", "all_pure", "all_noctx", "lodo"]:
        g = main_table[main_table.group == group]
        if len(g):
            per_policy.loc[GROUP_NAMES.get(group, group)] = g.set_index("dataset").agent_qoe
    best = base.groupby(["policy", "dataset"]).qoe.mean().unstack().max()
    regret = (best - per_policy).reindex(columns=DATASETS).clip(lower=0.0)
    regret["mean_regret"] = regret.mean(axis=1)
    regret["worst_regret"] = regret[DATASETS].max(axis=1)
    regret = regret.sort_values("mean_regret")
    regret.to_csv(os.path.join(args.results, "table_regret.csv"))
    summary["regret"] = regret.round(4).to_dict()

    with open(os.path.join(args.results, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    pd.set_option("display.width", 200)
    print(main_table[["group", "dataset", "agent_qoe", "agent_seed_std", "best_heuristic",
                      "best_heuristic_qoe", "mean_diff", "ci_low", "ci_high", "p_value",
                      "win_rate", "seeds_beating_best", "gain_vs_mpc"]].round(3).to_string(index=False))
    if delay_rows:
        print("\nmigration delay:\n", dt[["migration_delay", "dataset", "agent_qoe", "best_heuristic_qoe",
                                          "mean_diff", "agent_migrations_per_100",
                                          "ctx_joint_migrations"]].round(3).to_string(index=False))
    print("\nregret vs best heuristic per dataset:\n", regret.round(3))
    if matrix:
        print("\nsingle-source transfer (agent - best heuristic):\n", mat.round(3))


if __name__ == "__main__":
    main()
