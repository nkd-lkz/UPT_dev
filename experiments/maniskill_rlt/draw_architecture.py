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
        0.5,
        7.08,
        "RLT + FLARE-inspired future representations",
        fontsize=19,
        weight="bold",
    )
    ax.text(
        0.5,
        6.64,
        "Implemented research variant | Stage 1B: predict observed futures",
        fontsize=11,
        color="#425466",
    )
    box(0.5, 5.1, 2.6, 0.95, "Frozen Stage 1 VLA\nRGB + language + qpos", frozen)
    box(3.9, 5.1, 2.5, 0.95, "RL token z(t)\nproprioception p(t)", frozen)
    box(7.1, 5.1, 2.5, 0.95, "Current encoder E\ne(t), 128 dimensions", trainable)
    box(10.3, 5.1, 3.0, 0.95, "Actor / twin critics\noriginal inputs + e(t)", trainable)
    arrow((3.1, 5.58), (3.9, 5.58))
    arrow((6.4, 5.58), (7.1, 5.58))
    arrow((9.6, 5.58), (10.3, 5.58), "actor: stop-grad", small=True)
    box(
        0.5,
        2.8,
        2.6,
        1.15,
        "Executed action prefix\na(t) ... a(t+h-1)\nh = 1, 5, 10 offline",
        data,
    )
    box(
        4.0,
        2.8,
        3.3,
        1.15,
        "Transformer\n[e(t), action tokens, query(h)]\nthree prediction heads",
        trainable,
    )
    box(
        8.0,
        2.8,
        2.5,
        1.15,
        "Predicted future\nRL token + joint change\noptional: LN(z) + delta",
        trainable,
    )
    box(11.2, 2.8, 2.1, 1.15, "Future alignment\n+ joint-change loss", loss)
    arrow((3.1, 3.38), (4.0, 3.38))
    arrow((8.35, 5.1), (6.6, 3.95))
    arrow((7.3, 3.38), (8.0, 3.38))
    arrow((10.5, 3.38), (11.2, 3.38))
    box(
        8.4,
        1.0,
        4.9,
        0.95,
        "Actual future observation -> same frozen VLA\nstop-gradient target; no future input at inference",
        frozen,
    )
    arrow((12.2, 1.95), (12.2, 2.8))
    box(
        0.5,
        1.0,
        6.6,
        0.95,
        "Stage 2: real replay -> TD loss + future auxiliary loss\nActor: Q + BC; frozen VLA; optional action bound OFF",
        data,
    )
    arrow((11.2, 2.95), (7.3, 2.95), "prediction gradients", dashed=True, curve=-0.28)
    ax.text(
        0.5,
        0.42,
        "Future targets are frozen; online terminal rows are masked.",
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
