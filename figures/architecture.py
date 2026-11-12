import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "mathtext.fontset": "dejavuserif"})


def box(ax, x, y, w, h, text, fc="#ffffff", ec="#333333", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(ax, p, q, style="-|>", ls="-", color="#333333", text=None, tx=0, ty=0, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7, lw=0.8,
                                 linestyle=ls, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2 + tx, (p[1] + q[1]) / 2 + ty, text, ha="center",
                va="center", fontsize=6.5, color=color, style="italic")


def main(out="architecture"):
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.4, 7.6)
    ax.axis("off")

    data, ctrl = "#1f4e79", "#a33a3a"

    box(ax, 0.1, 2.3, 1.5, 1.0, "Client\n(player)")
    box(ax, 2.3, 2.3, 1.5, 1.0, "RAN\n$C(\\tau)$")
    box(ax, 4.4, 2.3, 1.4, 1.0, "SDN\nswitch")
    box(ax, 2.3, 3.95, 1.5, 0.95, "near-RT\nRIC")
    box(ax, 0.1, 5.9, 6.6, 1.1, "Joint controller (edge orchestrator)\n"
        "$s_t \\rightarrow a_t = (b_t, k_t)$", fc="#fbf1f1", ec=ctrl)
    edges = [("Near edge\n$S_1, d_1$", 4.3), ("Metro edge\n$S_2, d_2$", 2.6),
             ("Regional edge\n$S_3, d_3$", 0.9)]
    for label, y in edges:
        box(ax, 7.0, y, 2.8, 1.0, label, fc="#f2f6fa")

    arrow(ax, (1.6, 2.8), (2.3, 2.8), style="<|-|>", color=data)
    arrow(ax, (3.8, 2.8), (4.4, 2.8), style="<|-|>", color=data)
    for _, y in edges:
        arrow(ax, (5.8, 2.8), (7.0, y + 0.5), style="<|-|>", color=data)

    arrow(ax, (3.05, 3.3), (3.05, 3.95), color=ctrl, ls="--")
    arrow(ax, (3.05, 4.9), (3.05, 5.9), color=ctrl, ls="--")
    ax.text(3.25, 5.4, "radio\ncontext", ha="left", va="center", fontsize=6.5,
            color=ctrl, style="italic", linespacing=1.0)
    arrow(ax, (0.85, 5.9), (0.85, 3.3), color=ctrl)
    ax.text(1.05, 4.6, "bitrate\n$b_t$", ha="left", va="center", fontsize=6.5,
            color=ctrl, style="italic", linespacing=1.0)
    arrow(ax, (5.1, 5.9), (5.1, 3.3), color=ctrl)
    ax.text(5.3, 4.6, "steering\n$k_t$", ha="left", va="center", fontsize=6.5,
            color=ctrl, style="italic", linespacing=1.0)
    arrow(ax, (6.7, 6.2), (7.6, 5.3), color=ctrl)
    ax.text(7.05, 5.45, "migration", ha="right", va="center", fontsize=6.5,
            color=ctrl, style="italic")
    arrow(ax, (9.4, 5.3), (6.7, 6.75), color=ctrl, ls="--", rad=0.3)
    ax.text(8.6, 7.15, "edge load $u^k_t$", ha="center", va="center", fontsize=6.5,
            color=ctrl, style="italic")

    fig.tight_layout(pad=0.1)
    fig.savefig(f"{out}.pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None})
    fig.savefig(f"{out}.png", dpi=200, metadata={"Software": None})


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "architecture")
