# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Render an editable vector method diagram and a preview from the implemented graph."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main() -> None:
    """Write SVG, PDF and PNG beside this script, without accessing a GPU."""
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42}
    )
    fig, ax = plt.subplots(figsize=(14, 7.6))
    ax.set(xlim=(0, 14), ylim=(0, 7.6))
    ax.axis("off")
    frozen, trainable, data, loss = "#E5EFF8", "#DFF2E4", "#F0ECF8", "#FCE7DC"

    def box(x, y, w, h, text, color):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.04",
                facecolor=color,
                edgecolor="#425466",
                linewidth=1.2,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    def arrow(start, end, label="", dashed=False, curve=0, small=False):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=13,
                color="#485466",
                linewidth=1.2,
                linestyle="--" if dashed else "-",
                connectionstyle=f"arc3,rad={curve}",
            )
        )
        if label:
            ax.text(
                (start[0] + end[0]) / 2,
                (start[1] + end[1]) / 2 + (0.27 if small else -0.35),
                label,
                ha="center",
                fontsize=8,
                color="#485466",
            )

    ax.text(
        0.5, 7.08, "RLT + Zeva-inspired interaction memory", fontsize=19, weight="bold"
    )
    ax.text(
        0.5,
        6.64,
        "Implemented research variant | Completed interactions only",
        fontsize=11,
        color="#425466",
    )
    box(0.5, 5.1, 2.6, 0.95, "Frozen Stage 1 VLA\nRGB + language + qpos", frozen)
    box(3.8, 5.1, 2.6, 0.95, "RL token + qpos\nreference action chunk", frozen)
    box(
        7.2,
        5.1,
        2.6,
        0.95,
        "Actor / twin critics\noriginal inputs + context",
        trainable,
    )
    box(10.5, 5.1, 2.8, 0.95, "Route and execute\n10-step action chunk", data)
    arrow((3.1, 5.58), (3.8, 5.58))
    arrow((6.4, 5.58), (7.2, 5.58))
    arrow((9.8, 5.58), (10.5, 5.58))
    box(
        10.5,
        2.8,
        2.8,
        1.15,
        "Completed interaction\nstart qpos, executed actions\nactual change, terminal flags",
        data,
    )
    box(
        6.8,
        2.8,
        2.8,
        1.15,
        "Per-environment memory\nrecent 4 + archive 32\nretrieve 4 older records",
        data,
    )
    box(
        3.2,
        2.8,
        2.8,
        1.15,
        "MLP + attention reader\nquery = current raw qpos\n64-dimensional context",
        trainable,
    )
    arrow((11.9, 5.1), (11.9, 3.95))
    arrow((10.5, 3.38), (9.6, 3.38))
    arrow((6.8, 3.38), (6.0, 3.38))
    arrow((5.3, 3.95), (7.6, 5.1), "actor: stop-grad", small=True)
    box(
        0.5,
        1.0,
        5.8,
        0.95,
        "Replay stores decision-time raw memory snapshots\nNo future records; empty memory returns zero",
        data,
    )
    box(
        7.2,
        1.0,
        6.1,
        0.95,
        "Stage 2: critic TD loss trains reader + critic\nActor: original Q + BC; Stage 1 unchanged",
        loss,
    )
    arrow((9.5, 1.95), (4.6, 2.8), "TD gradients", dashed=True)
    ax.text(
        0.5,
        0.42,
        "Memory is read before action; evidence is appended after execution.",
        fontsize=10,
        color="#425466",
    )
    ax.text(
        0.5,
        0.10,
        "Blue: frozen / targets     Green: trainable     Purple: observations / execution     Dashed: gradient path",
        fontsize=9,
        color="#425466",
    )
    output = Path(__file__).parent / "figures"
    output.mkdir(exist_ok=True)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(
            output / f"architecture.{suffix}",
            dpi=220,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
