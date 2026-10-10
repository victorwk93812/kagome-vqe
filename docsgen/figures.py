"""Matplotlib figures, saved as PDF for the report and SVG for the deck.

Every figure reads its data from :mod:`docsgen.physics`, :mod:`docsgen.runs` or the package;
nothing here types a number by hand. Colours follow :mod:`docsgen.style`: accent for the point
of the figure, derived/side-b for the two ansatzes (HVA purple, RVB teal), one fixed hue per
gradient method.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Polygon

from docsgen import geometry, physics, runs, style
from docsgen.paths import REPORT_FIGURES, SLIDES_MEDIA
from kagome_vqe.ansatz import ring_bonds
from kagome_vqe.lattice import KAGOME_HAMILTONIAN_CYCLE, KAGOME_UNIT_CELL_EDGES
from kagome_vqe.results import LoadedRun

__all__ = ["ALL", "write_all"]

ANSATZ_COLORS = {"hva": style.DERIVED, "rvb": style.SIDE_B}
COMBOS = ("hva_spsa", "hva_adam", "rvb_spsa", "rvb_adam")
METHODS = tuple(style.METHOD_COLORS)
MARKERS = dict(zip(METHODS, ("o", "s", "^", "D", "v"), strict=True))


def _save(fig: Figure, name: str) -> list[Path]:
    REPORT_FIGURES.mkdir(parents=True, exist_ok=True)
    SLIDES_MEDIA.mkdir(parents=True, exist_ok=True)
    out = [REPORT_FIGURES / f"{name}.pdf", SLIDES_MEDIA / f"{name}.svg"]
    for path in out:
        fig.savefig(
            path, metadata={"Date": None} if path.suffix == ".svg" else {"CreationDate": None}
        )
    plt.close(fig)
    return out


# -- lattice -----------------------------------------------------------------------------------


def _draw_lattice(
    ax: Axes,
    *,
    highlight: list[tuple[int, int]] = (),
    color: str = style.ACCENT,
    labels: dict[int, str] | None = None,
    shade: bool = True,
) -> None:
    pos = geometry.positions()
    if shade:
        for tri in geometry.triangles():
            ax.add_patch(
                Polygon([pos[s] for s in tri], closed=True, fc=style.MUTED, alpha=0.12, lw=0)
            )
    hl = {frozenset(b) for b in highlight}
    for a, b in KAGOME_UNIT_CELL_EDGES:
        on = frozenset((a, b)) in hl
        ax.plot(
            *zip(pos[a], pos[b], strict=True),
            color=color if on else style.MUTED,
            lw=4.0 if on else 1.3,
            solid_capstyle="round",
            zorder=2 if on else 1,
        )
    for s, (x, y) in pos.items():
        ax.add_patch(plt.Circle((x, y), 0.17, fc="white", ec=style.INK_2, lw=0.9, zorder=3))
        ax.text(
            x,
            y,
            labels[s] if labels else str(s),
            ha="center",
            va="center",
            fontsize=8.5,
            color=style.INK,
            zorder=4,
        )
    ax.set_aspect("equal")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.axis("off")


def fig_lattice() -> Figure:
    """Lattice, Hamiltonian cycle, and the two dimer coverings."""
    f = physics.facts()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
    _draw_lattice(axes[0])
    axes[0].set_title("(a) 12 sites, 18 bonds, 6 triangles")
    cycle = KAGOME_HAMILTONIAN_CYCLE
    ring = [(cycle[k], cycle[(k + 1) % 12]) for k in range(12)]
    qubit = {s: f"q{k}" for k, s in enumerate(cycle)}
    _draw_lattice(axes[1], highlight=ring, color=style.DERIVED, labels=qubit)
    axes[1].set_title("(b) ring order = Hamiltonian cycle")
    for ax, matching, tag in zip(axes[2:], f.matchings, "cd", strict=True):
        _draw_lattice(ax, highlight=list(matching), color=style.ACCENT)
        ax.set_title(f"({tag}) dimer covering, E = {-3 * len(matching)}")
    fig.tight_layout()
    return fig


# -- circuits ----------------------------------------------------------------------------------


def _ring_circuit(ansatz: str) -> Figure:
    n = 12
    odd, even = ring_bonds(n)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ys = {q: n - 1 - q for q in range(n)}
    x_end = 9.4
    for q in range(n):
        ax.plot([0.2, x_end], [ys[q]] * 2, color=style.INK_2, lw=0.8, zorder=0)
        ax.text(
            0.05,
            ys[q],
            f"$q_{{{q}}}$ · site {KAGOME_HAMILTONIAN_CYCLE[q]}",
            ha="right",
            va="center",
            fontsize=8.5,
        )

    def box(x: float, q0: int, q1: int, text: str, color: str, w: float = 1.3) -> None:
        y0, y1 = ys[q1] - 0.32, ys[q0] + 0.32
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y0),
                w,
                y1 - y0,
                boxstyle="round,pad=0,rounding_size=0.12",
                fc="white",
                ec=color,
                lw=1.6,
                zorder=2,
            )
        )
        ax.text(
            x, (y0 + y1) / 2, text, ha="center", va="center", fontsize=8, color=style.INK, zorder=3
        )

    for a, b in even:
        box(1.4, a, b, r"$|s\rangle$", style.ACCENT, w=1.0)
    label = (lambda k: rf"$\phi_{{{k}}}$") if ansatz == "rvb" else (lambda _k: r"$\phi$")
    x_odd, x_even = 4.0, 6.8
    for k, (a, b) in enumerate(odd):
        if b == 0:  # wrap-around bond (11, 0): two half boxes joined by a dashed line
            box(x_odd, 0, 0, label(k), style.DERIVED, w=1.3)
            box(x_odd, 11, 11, label(k), style.DERIVED, w=1.3)
            ax.plot(
                [x_odd + 0.75] * 2,
                [ys[11], ys[0]],
                color=style.DERIVED,
                lw=1.0,
                ls=(0, (4, 3)),
                zorder=1,
            )
        else:
            box(x_odd, a, b, label(k), style.DERIVED)
    even_label = (lambda k: rf"$\phi_{{{k + 6}}}$") if ansatz == "rvb" else (lambda _k: r"$\gamma$")
    for k, (a, b) in enumerate(even):
        box(x_even, a, b, even_label(k), style.SIDE_B)
    gate = (
        r"eSWAP$(\phi)=e^{-i\phi\,\mathrm{SWAP}/2}$"
        if ansatz == "rvb"
        else r"$e^{-i\theta(XX+YY+ZZ)/2}$"
    )
    ax.text(1.4, n - 0.2, "singlets", ha="center", fontsize=8.5, color=style.INK_2)
    ax.text(x_odd, n - 0.2, "odd ring bonds", ha="center", fontsize=8.5, color=style.INK_2)
    ax.text(x_even, n - 0.2, "even ring bonds", ha="center", fontsize=8.5, color=style.INK_2)
    ax.text(x_end, -1.1, f"each box: {gate}", ha="right", fontsize=8.5, color=style.INK_2)
    ax.set_xlim(-2.0, x_end + 0.6)
    ax.set_ylim(-1.5, n + 0.3)
    ax.axis("off")
    return fig


def fig_circuit_rvb() -> Figure:
    """RVB: singlets on even ring bonds, then one eSWAP per ring bond with its own angle."""
    return _ring_circuit("rvb")


def fig_circuit_hva() -> Figure:
    """HVA: singlets, then shared-angle Heisenberg evolution on odd and even ring bonds."""
    return _ring_circuit("hva")


# -- noise floor -------------------------------------------------------------------------------


def fig_noise_floor() -> Figure:
    """Energy at the exact parameters with each noise channel switched on separately."""
    nf = physics.noise_floor()
    settings = [
        ("noise_free", "noise-free\n(sampled)"),
        ("readout", "readout\nonly"),
        ("gate", "gate + T1/T2\nonly"),
        ("full", "full\nmodel"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    width = 0.36
    x = np.arange(len(settings))
    for i, ansatz in enumerate(("hva", "rvb")):
        vals = [-nf["energies"][f"{ansatz}_{k}"] for k, _ in settings]
        bars = ax.bar(
            x + (i - 0.5) * width,
            vals,
            width * 0.92,
            color=ANSATZ_COLORS[ansatz],
            label=ansatz.upper(),
            zorder=2,
        )
        for rect, v in zip(bars, vals, strict=True):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                v + 0.2,
                f"{-v:.2f}",
                ha="center",
                fontsize=7.5,
                color=style.INK,
            )
    ax.axhline(-physics.facts().spectrum[0], color=style.ACCENT, ls="--", lw=1.4, zorder=3)
    ax.text(
        len(settings) - 0.5,
        -physics.facts().spectrum[0] + 0.25,
        "exact |E| = 18",
        ha="right",
        fontsize=8,
        color=style.ACCENT,
    )
    ax.set_xticks(x, [label for _, label in settings])
    ax.set_ylabel("|E| at exact parameters")
    ax.set_ylim(0, 20.5)
    ax.legend(loc="lower left")
    ax.set_title(f"FakeGuadalupeV2, {nf['shots']} shots per basis, 66 CX")
    fig.tight_layout()
    return fig


# -- M0 ----------------------------------------------------------------------------------------


def _iter_trace(run: LoadedRun) -> np.ndarray:
    return np.asarray(run.energies_per_iteration(), dtype=float)


def fig_m0_before_after() -> Figure:
    """Noisy convergence before and after M0, per combination, with the re-measured final."""
    before = {r.name.split("-", 1)[1]: r for r in runs.loaded_runs("fake_guadalupe")}
    after = {r.name.split("-", 1)[1]: r for r in runs.loaded_runs("fake_guadalupe_m0")}
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    for ax, combo in zip(axes, COMBOS, strict=True):
        b, a = _iter_trace(before[combo]), _iter_trace(after[combo])
        ax.plot(
            np.arange(1, len(b) + 1), b, color=style.MUTED, lw=1.6, label="before M0 (no sampling)"
        )
        ax.plot(np.arange(1, len(a) + 1), a, color=style.ACCENT, lw=1.6, label="after M0 (sampled)")
        final = float(after[combo].result["final_energy"])
        ax.scatter(
            [len(a)],
            [final],
            s=36,
            color=style.ACCENT_INK,
            edgecolor="white",
            linewidth=1.5,
            zorder=4,
            label="re-measured final",
        )
        ax.set_title(combo.replace("_", " + ").upper())
        ax.set_xlabel("iteration")
    axes[0].set_ylabel("energy (best of iteration)")
    axes[0].legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    return fig


# -- M1 ----------------------------------------------------------------------------------------


def _ideal_curves(group: list[LoadedRun]) -> tuple[np.ndarray, np.ndarray]:
    """Error of the noise-free energy against shots on a common grid (one row per seed).

    Same metric as :func:`kagome_vqe.plotting.plot_ideal_vs_shots`: the error after spending
    ``s`` shots is the error of the last iterate reached by ``s``.
    """
    traces: list[pd.DataFrame] = [r.ideal_trace for r in group if r.ideal_trace is not None]
    exact = float(group[0].result["exact_energy"])
    lo = min(float(t["shots"].iloc[1]) for t in traces)
    hi = max(float(t["shots"].iloc[-1]) for t in traces)
    grid = np.geomspace(max(lo, 1.0), hi, 80)
    rows = []
    for t in traces:
        err = np.maximum((t["ideal_energy"].to_numpy(float) - exact) / abs(exact), 1e-7)
        idx = np.searchsorted(t["shots"].to_numpy(float), grid, side="right") - 1
        rows.append(err[np.clip(idx, 0, len(err) - 1)])
    return grid, np.vstack(rows)


def _method_of(run: LoadedRun) -> str:
    combo = run.combo.split("-", 1)[1]  # "rvb_adam_forward"
    return combo.split("_", 1)[1]


def fig_m1_ideal_vs_shots() -> Figure:
    """Median and min-max band of the noise-free relative error against shots, per method."""
    groups: dict[tuple[str, str], list[LoadedRun]] = {}
    for run in runs.loaded_runs("m1_ideal"):
        groups.setdefault((str(run.result["ansatz"]), _method_of(run)), []).append(run)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, ansatz in zip(axes, ("hva", "rvb"), strict=True):
        for method in METHODS:
            grid, curves = _ideal_curves(groups[(ansatz, method)])
            color = style.METHOD_COLORS[method]
            ax.plot(
                grid,
                np.median(curves, axis=0),
                color=color,
                lw=2,
                label=style.METHOD_LABELS[method],
                marker=MARKERS[method],
                markevery=12,
                ms=5,
            )
            ax.fill_between(
                grid, curves.min(axis=0), curves.max(axis=0), color=color, alpha=0.12, lw=0
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-6)
        ax.set_title(
            f"{ansatz.upper()}  ({len(groups[(ansatz, METHODS[0])])} seeds, shot noise only)"
        )
        ax.set_xlabel("shots spent (all bases)")
    axes[0].set_ylabel("noise-free relative error of parameters")
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    return fig


def fig_m1_final() -> Figure:
    """Final noise-free relative error per method: one dot per seed, a bar at the mean."""
    agg = runs.aggregate("m1_ideal", "ideal_relative_error")
    noisy = runs.by_name("m1_noisy")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharex=True)
    for ax, ansatz in zip(axes, ("hva", "rvb"), strict=True):
        for i, method in enumerate(METHODS):
            a = agg[f"{ansatz}_{method}"]
            y = np.full(a.n, len(METHODS) - 1 - i) + np.linspace(-0.12, 0.12, a.n)
            color = style.METHOD_COLORS[method]
            ax.scatter(
                np.maximum(a.values, 1e-6) * 100,
                y,
                s=22,
                color=color,
                marker=MARKERS[method],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            ax.plot(
                [a.mean * 100] * 2, [y[0] - 0.2, y[-1] + 0.2], color=style.INK, lw=1.6, zorder=4
            )
            key = f"rvb_{method}"
            if ansatz == "rvb" and key in noisy:
                ax.scatter(
                    [noisy[key]["ideal_relative_error"] * 100],
                    [len(METHODS) - 1 - i + 0.33],
                    s=40,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.6,
                    marker=MARKERS[method],
                    zorder=3,
                )
        ax.set_yticks(range(len(METHODS)), [style.METHOD_LABELS[m] for m in reversed(METHODS)])
        ax.set_xscale("log")
        ax.set_xlabel("noise-free relative error of final parameters (%)")
        ax.set_title(ansatz.upper())
        ax.grid(axis="y", visible=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.text(
        0.99,
        0.01,
        "filled marks: one per seed, shot noise only; black bar: mean over seeds;  "
        "hollow mark (RVB): one seed with FakeGuadalupeV2 noise",
        ha="right",
        va="bottom",
        fontsize=8,
        color=style.INK_2,
    )
    return fig


# -- landscape ---------------------------------------------------------------------------------


def fig_hva_landscape() -> Figure:
    """HVA energy landscape for the two site orders, minimum marked."""
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharey=True)
    images = []
    for ax, order in zip(axes, ("cycle", "identity"), strict=True):
        axis, energy = physics.hva_landscape(order)
        images.append(
            ax.imshow(
                energy,
                origin="lower",
                extent=(axis[0], axis[-1], axis[0], axis[-1]),
                cmap="Purples_r",
                vmin=-18,
                vmax=18,
                aspect="equal",
            )
        )
        # the minimum is degenerate; mark the grid minimiser closest to the origin
        ii, jj = np.nonzero(energy <= energy.min() + 1e-9)
        k = int(np.argmin(axis[ii] ** 2 + axis[jj] ** 2))
        i, j = ii[k], jj[k]
        ax.scatter(
            [axis[j]],
            [axis[i]],
            s=60,
            marker="*",
            color=style.ACCENT,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            f"min {energy.min():.2f}".replace("-", "\u2212"),
            (axis[j], axis[i]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8.5,
            color=style.INK,
            bbox={"fc": "white", "ec": "none", "alpha": 0.8, "pad": 1.5},
        )
        ax.set_title("Hamiltonian-cycle order" if order == "cycle" else "identity order")
        ax.set_xlabel(r"odd-bond angle $\phi$")
        ax.grid(visible=False)
    axes[0].set_ylabel(r"even-bond angle $\gamma$")
    fig.colorbar(images[0], ax=axes, shrink=0.85, label="energy")
    return fig


ALL: dict[str, Callable[[], Figure]] = {
    "lattice": fig_lattice,
    "circuit_rvb": fig_circuit_rvb,
    "circuit_hva": fig_circuit_hva,
    "hva_landscape": fig_hva_landscape,
    "noise_floor": fig_noise_floor,
    "m0_before_after": fig_m0_before_after,
    "m1_ideal_vs_shots": fig_m1_ideal_vs_shots,
    "m1_final": fig_m1_final,
}


def write_all(only: set[str] | None = None) -> list[Path]:
    """Render every figure (or the named subset) to PDF and SVG."""
    style.apply()
    out: list[Path] = []
    for name, make in ALL.items():
        if only is None or name in only:
            out += _save(make(), name)
    return out
