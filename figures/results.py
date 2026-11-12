import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
HATCHES = ["", "////", "....", "xxxx"]
LABELS = {"hsdpa": "HSDPA\n3G", "fcc": "FCC\nfixed", "oboe": "Oboe\nmixed", "puffer": "Puffer\n2021-22",
          "5g": "5G\nmeasured"}
ORDER = ["hsdpa", "fcc", "oboe", "puffer", "5g"]
NO_META = {"Creator": None, "Producer": None, "CreationDate": None}


def save(fig, path):
    fig.savefig(path + ".pdf", metadata=NO_META)
    fig.savefig(path + ".png", dpi=220, metadata={"Software": None})
    plt.close(fig)


def gain_bars(main, groups, names, path, floor=-1.5):
    fig, ax = plt.subplots(figsize=(3.45, 1.8))
    x = np.arange(len(ORDER))
    width = 0.8 / len(groups)
    for i, (g, name) in enumerate(zip(groups, names)):
        sub = main[main.group == g].set_index("dataset").reindex(ORDER)
        pos = x - 0.4 + width * (i + 0.5)
        err = np.vstack([sub.mean_diff - sub.ci_low, sub.ci_high - sub.mean_diff])
        cut = sub.mean_diff.values < floor
        err[:, cut] = 0.0
        ax.bar(pos, np.maximum(sub.mean_diff.values, floor), width * 0.88, color=SERIES[i], hatch=HATCHES[i],
               edgecolor="white", linewidth=0.0, label=name, zorder=2)
        ax.errorbar(pos, sub.mean_diff, yerr=err, fmt="none", ecolor="#333333",
                    elinewidth=0.6, capsize=1.6, zorder=3)
        for xp, v in zip(pos[cut], sub.mean_diff.values[cut]):
            ax.text(xp + width * 0.55, floor + 0.06, f"{v:.1f}".replace("-", "\u2212"),
                    ha="left", va="bottom", fontsize=6.2, color="#333333")
    lo = min(floor, float(np.nanmin(main[main.group.isin(groups)].ci_low.clip(lower=floor))))
    hi = float(main[main.group.isin(groups)].ci_high.max())
    ax.set_ylim(lo, max(0.25, hi + 0.1))
    ax.axhline(0, color="#333333", lw=0.7, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[d] for d in ORDER])
    ax.set_ylabel("QoE minus best heuristic")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=1, loc="lower left", frameon=False, handlelength=1.4, handletextpad=0.5)
    fig.tight_layout(pad=0.2)
    save(fig, path)


def delay_lines(delay, path):
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.6))
    agg = delay.groupby("migration_delay").agg(
        agent_mig=("agent_migrations_per_100", "mean"),
        ctx_mig=("ctx_joint_migrations", "mean"),
        gain=("mean_diff", "mean")).reset_index()
    d = agg.migration_delay
    axes[0].plot(d, agg.agent_mig, "-o", color=SERIES[0], lw=1.4, ms=3.5, label="Learned")
    axes[0].plot(d, agg.ctx_mig, "--s", color=SERIES[1], lw=1.4, ms=3.5, label="Ctx-Joint")
    axes[0].set_xlabel("migration delay $\\delta$ (s)")
    axes[0].set_ylabel("migrations per 100")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].plot(d, agg.gain, "-o", color=SERIES[0], lw=1.4, ms=3.5)
    axes[1].axhline(0, color="#333333", lw=0.7)
    axes[1].set_xlabel("migration delay $\\delta$ (s)")
    axes[1].set_ylabel("QoE minus best heur.")
    fig.tight_layout(pad=0.2, w_pad=0.8)
    save(fig, path)


def transfer_heatmap(matrix, path):
    m = matrix.reindex(index=ORDER, columns=ORDER)
    lim = float(np.nanmax(np.abs(m.values)))
    fig, ax = plt.subplots(figsize=(2.6, 2.1))
    im = ax.imshow(m.values, cmap="RdBu", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(ORDER)))
    ax.set_yticks(range(len(ORDER)))
    short = {"hsdpa": "HSDPA", "fcc": "FCC", "oboe": "Oboe", "puffer": "Puffer", "5g": "5G"}
    ax.set_xticklabels([short[d] for d in ORDER])
    ax.set_yticklabels([short[d] for d in ORDER])
    ax.set_xlabel("test dataset")
    ax.set_ylabel("training dataset")
    ax.grid(False)
    for i in range(len(ORDER)):
        for j in range(len(ORDER)):
            v = m.values[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.6 * lim else "#111111")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.4)
    fig.tight_layout(pad=0.2)
    save(fig, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="../paper")
    p.add_argument("--groups", default="all,lodo,all_pure")
    p.add_argument("--names", default="Learning-augmented,Unseen dataset,From scratch")
    args = p.parse_args()
    main_t = pd.read_csv(os.path.join(args.results, "table_main.csv"))
    groups = args.groups.split(",")
    gain_bars(main_t, groups, args.names.split(","), os.path.join(args.out, "fig_gain"))
    if os.path.exists(os.path.join(args.results, "table_delay.csv")):
        delay_lines(pd.read_csv(os.path.join(args.results, "table_delay.csv")),
                    os.path.join(args.out, "fig_delay"))
    tm = os.path.join(args.results, "table_transfer_matrix.csv")
    if os.path.exists(tm):
        transfer_heatmap(pd.read_csv(tm, index_col=0), os.path.join(args.out, "fig_transfer"))


if __name__ == "__main__":
    main()
