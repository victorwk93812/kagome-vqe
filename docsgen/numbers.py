r"""Every number quoted in the report or the deck, with its provenance.

:func:`collect` returns a registry ``key -> Number``. Writers turn it into
``report/tables/numbers.tex`` (one ``\V<key>`` macro per entry, e.g. ``\VExactE``) and
``slides/data/numbers.json`` (``{"ExactE": {"value": -18.0, "text": "−18.000", ...}}``).
Keys are letters only so they are valid TeX control-sequence names.

Two kinds of source exist. ``computed`` numbers are recomputed here from the package or read
from ``results/``. ``quoted`` numbers come from the repository's own prose (``docs/roadmap.md``)
because the code that produced them no longer exists (the pre-M0 seed bug); each names its
source and commit, and tests check they are not silently mixed with computed ones.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from docsgen import physics, runs
from docsgen.paths import REPORT_TABLES, ROOT, SLIDES_DATA
from kagome_vqe.results import load_run

__all__ = ["Number", "collect", "write_json", "write_tex"]

Kind = Literal["num", "int", "pct", "text"]
_KEY = re.compile(r"^[A-Za-z]+$")

METHODS = ("adam_forward", "adam_central", "adam_spsa", "adam_parameter_shift", "spsa")
METHOD_KEYS = {
    "adam_forward": "Fwd",
    "adam_central": "Cen",
    "adam_spsa": "Hyb",
    "adam_parameter_shift": "Ps",
    "spsa": "Spsa",
}
ANSATZ_KEYS = {"hva": "Hva", "rvb": "Rvb"}
STACKS = (
    "none",
    "readout",
    "symmetry",
    "readout+symmetry",
    "zne",
    "readout+zne",
    "readout+symmetry+zne",
    "twin",
    "readout+twin",
    "readout+symmetry+twin",
)
STACK_KEYS = {
    "none": "Raw",
    "readout": "Ro",
    "symmetry": "Sym",
    "readout+symmetry": "RoSym",
    "zne": "Zne",
    "readout+zne": "RoZne",
    "readout+symmetry+zne": "RoSymZne",
    "twin": "Twin",
    "readout+twin": "RoTwin",
    "readout+symmetry+twin": "RoSymTwin",
}
POINT_KEYS = {"m2-evaluation-0": "Zero", "m2-evaluation-0p3": "Pt"}
EXACT_MODE_KEYS = {
    "raw": "Raw",
    "readout": "Ro",
    "symmetry": "Sym",
    "readout_symmetry": "RoSym",
    "correct_then_filter": "Ctf",
}
COMBO_KEYS = {
    "rvb_spsa": "RvbSpsa",
    "rvb_adam": "RvbAdam",
    "hva_spsa": "HvaSpsa",
    "hva_adam": "HvaAdam",
}


@dataclass(frozen=True)
class Number:
    """One quoted value.

    Attributes:
        value: The number (``nan`` for ``text`` kind).
        kind: ``num`` (fixed decimals), ``int``, ``pct`` (value is a fraction, shown in %), or
            ``text`` (a pre-formatted string in ``text``).
        decimals: Decimals shown for ``num`` and ``pct``.
        source: Where the value comes from (a results path, a function, or a document).
        quoted: True when copied from prose rather than recomputed.
        text: Display string for ``text`` kind.
    """

    value: float
    kind: Kind = "num"
    decimals: int = 3
    source: str = ""
    quoted: bool = False
    text: str = ""

    def plain(self) -> str:
        """Display string with a typographic minus (for HTML)."""
        if self.kind == "text":
            return self.text
        if self.kind == "int":
            s = f"{round(self.value)}"
        elif self.kind == "pct":
            s = f"{100 * self.value:.{self.decimals}f} %"
        else:
            s = f"{self.value:.{self.decimals}f}"
        if s.lstrip("-").strip("0. %") == "":
            s = s.lstrip("-")  # no negative zero
        return s.replace("-", "−")

    def math(self) -> str:
        """Display string for KaTeX in the deck (ASCII minus, escaped percent)."""
        if self.kind == "text":
            return f"\\text{{{self.text}}}"
        return self.plain().replace("\u2212", "-").replace(" %", "\\%")

    def tex(self) -> str:
        """Display string for LaTeX (math-mode minus, thin space before %)."""
        if self.kind == "text":
            return self.text
        if self.kind == "int":
            body = f"{round(self.value)}"
        elif self.kind == "pct":
            body = f"{100 * self.value:.{self.decimals}f}\\,\\%"
        else:
            body = f"{self.value:.{self.decimals}f}"
        if body.startswith("-") and body.lstrip("-").strip("0.\\,%") == "":
            body = body.lstrip("-")  # no negative zero
        return f"\\ensuremath{{{body}}}"


class Registry(dict[str, Number]):
    """``dict`` with a checked :meth:`add`."""

    def add(self, key: str, value: float, **kwargs: object) -> None:
        """Register ``key``; keys are unique and letters-only."""
        if not _KEY.match(key):
            msg = f"number key {key!r} must be letters only (it becomes a TeX macro)"
            raise ValueError(msg)
        if key in self:
            msg = f"duplicate number key {key!r}"
            raise KeyError(msg)
        self[key] = Number(float(value), **kwargs)  # type: ignore[arg-type]


def _per_iteration(run_path: str) -> tuple[float, int, int]:
    """(energies per iteration, maxiter, shots per energy) of one recorded run."""
    run = load_run(ROOT / run_path)
    maxiter = int(run.config.optimizer.maxiter)
    nfev = int(run.result["nfev"])
    shots = int(run.config.backend.shots)
    return (nfev - 1) / maxiter, maxiter, shots


def collect(*, recompute: bool = False) -> Registry:
    """Build the registry (reads ``results/``; simulator studies come from the cache)."""
    reg = Registry()
    _exact(reg)
    _noise(reg, recompute=recompute)
    _m0(reg)
    _m1(reg)
    _m2_fixed_point(reg)
    _m2_vqe(reg)
    _m2_exact(reg, recompute=recompute)
    return reg


def _exact(reg: Registry) -> None:
    f = physics.facts()
    src = "docsgen.physics.facts"
    reg.add("ExactE", f.spectrum[0], decimals=3, source=src)
    reg.add("ExactEone", f.spectrum[1], decimals=3, source=src)
    reg.add("ExactEtwo", f.spectrum[2], decimals=4, source=src)
    reg.add("ExactGap", f.spectrum[2] - f.spectrum[0], decimals=4, source=src)
    reg.add("NumSites", 12, kind="int", source=src)
    reg.add("NumBonds", f.num_bonds, kind="int", source=src)
    reg.add("NumTerms", f.num_terms, kind="int", source=src)
    reg.add("NumMatchings", len(f.matchings), kind="int", source=src)
    reg.add("SzSector", math.comb(12, 6), kind="int", source="binomial(12, 6)")
    reg.add("RingFracCycle", f.ring_fraction_cycle, kind="pct", decimals=0, source=src)
    reg.add("RingFracIdentity", f.ring_fraction_identity, kind="pct", decimals=0, source=src)
    reg.add("SingletIdentity", f.singlet_bonds_identity, kind="int", source=src)
    for key, value in f.e_zero.items():
        ansatz, order = key.split("_")
        reg.add(f"Ezero{ANSATZ_KEYS[ansatz]}{order.capitalize()}", value, decimals=1, source=src)
    for ansatz in ("hva", "rvb"):
        a = ANSATZ_KEYS[ansatz]
        reg.add(f"Params{a}", f.num_parameters[ansatz], kind="int", source=src)
        reg.add(f"Cx{a}", f.cx[ansatz], kind="int", source=src)
        reg.add(f"Cx{a}Unshared", f.cx[f"{ansatz}_unshared"], kind="int", source=src)
        reg.add(f"Cx{a}NoLayout", f.cx[f"{ansatz}_nolayout"], kind="int", source=src)
        reg.add(f"Depth{a}", f.depth[ansatz], kind="int", source=src)
    reg.add(
        "HvaMinIdentity",
        physics.hva_minimum("identity"),
        decimals=2,
        source="docsgen.physics.hva_minimum",
    )
    reg.add("CoveringOverlap", f.covering_overlap, decimals=5, source=src)
    reg.add("MeasGroups", f.measurement_groups, kind="int", source=src)
    # parameter-shift cost: two energies per gate occurrence
    reg.add("PsEnergiesRvb", 2 * f.num_parameters["rvb"], kind="int", source="2 x 12 eSWAPs")
    reg.add(
        "PsEnergiesHva", 2 * 18 * f.num_parameters["hva"], kind="int", source="2 x 36 rotations"
    )


def _noise(reg: Registry, *, recompute: bool) -> None:
    nf = physics.noise_floor(recompute=recompute)
    src = "docsgen.physics.noise_floor (cache/noise_floor.json)"
    reg.add("NoiseShots", nf["shots"], kind="int", source=src)
    for ansatz in ("hva", "rvb"):
        a = ANSATZ_KEYS[ansatz]
        e = nf["energies"]
        for label, key in (
            ("noise_free", "Ideal"),
            ("readout", "Readout"),
            ("gate", "Gate"),
            ("full", "Full"),
        ):
            reg.add(f"Floor{a}{key}", e[f"{ansatz}_{label}"], decimals=2, source=src)
        ideal = e[f"{ansatz}_noise_free"]
        r_ro = e[f"{ansatz}_readout"] / ideal
        r_g = e[f"{ansatz}_gate"] / ideal
        reg.add(f"Keep{a}Readout", r_ro, kind="pct", decimals=1, source=src)
        reg.add(f"Keep{a}Gate", r_g, kind="pct", decimals=1, source=src)
        reg.add(f"Keep{a}Product", ideal * r_ro * r_g, decimals=2, source=src)
    bc = physics.bound_compilation(recompute=recompute)
    src = "docsgen.physics.bound_compilation (cache/bound_compilation.json)"
    reg.add("BoundTheta", bc["theta"], decimals=1, source=src)
    for ansatz in ("hva", "rvb"):
        a = ANSATZ_KEYS[ansatz]
        for label, word in (("parametric", "Param"), ("bound", "Bound")):
            row = bc["rows"][f"{ansatz}_{label}"]
            reg.add(f"{word}Cx{a}", row["cx"], kind="int", source=src)
            reg.add(f"{word}Depth{a}", row["depth"], kind="int", source=src)
            reg.add(f"{word}Keep{a}", row["retained"], kind="pct", decimals=1, source=src)
            reg.add(f"{word}Noisy{a}", row["noisy"], decimals=2, source=src)
        reg.add(f"BoundExact{a}", bc["rows"][f"{ansatz}_bound"]["exact"], decimals=2, source=src)
    sp = physics.shot_spread(recompute=recompute)
    src = "docsgen.physics.shot_spread (cache/shot_spread.json)"
    reg.add("SpreadRepeats", sp["repeats"], kind="int", source=src)
    for shots, word in (("1024", "Low"), ("4096", "High")):
        reg.add(f"Spread{word}Real", sp["real"][f"start_{shots}"], decimals=3, source=src)
        reg.add(f"Spread{word}Reported", sp["reported"][f"start_{shots}"], decimals=2, source=src)
        reg.add(f"Spread{word}Zero", sp["real"][f"zero_{shots}"], decimals=3, source=src)
    reg.add(
        "SpreadRatio",
        sp["reported"]["start_1024"] / sp["real"]["start_1024"],
        decimals=0,
        source=src,
    )


def _m0(reg: Registry) -> None:
    pre = runs.by_name("fake_guadalupe")
    post = runs.by_name("fake_guadalupe_m0")
    ideal = runs.by_name("statevector_m0")
    for combo, c in COMBO_KEYS.items():
        reg.add(
            f"PreFinal{c}",
            pre[combo]["energy"],
            decimals=2,
            source="results/fake_guadalupe-summary",
        )
        reg.add(
            f"PreBest{c}",
            pre[combo]["best_energy"],
            decimals=2,
            source="results/fake_guadalupe-summary",
        )
        reg.add(
            f"PostFinal{c}",
            post[combo]["final_energy"],
            decimals=2,
            source="results/fake_guadalupe_m0-summary",
        )
        reg.add(
            f"PostErr{c}",
            post[combo]["final_relative_error"],
            kind="pct",
            decimals=1,
            source="results/fake_guadalupe_m0-summary",
        )
        reg.add(
            f"PostBest{c}",
            post[combo]["best_energy"],
            decimals=2,
            source="results/fake_guadalupe_m0-summary",
        )
        reg.add(
            f"SvFinal{c}",
            ideal[combo]["final_energy"],
            decimals=4,
            source="results/statevector_m0-summary",
        )
        reg.add(
            f"SvErr{c}",
            ideal[combo]["final_relative_error"],
            kind="pct",
            decimals=3,
            source="results/statevector_m0-summary",
        )
    post_vals = [post[c]["final_energy"] for c in COMBO_KEYS]
    reg.add("PostFinalMin", min(post_vals), decimals=2, source="results/fake_guadalupe_m0-summary")
    reg.add("PostFinalMax", max(post_vals), decimals=2, source="results/fake_guadalupe_m0-summary")
    reg.add(
        "PostFinalShots",
        post["hva_spsa"]["final_shots"],
        kind="int",
        source="results/fake_guadalupe_m0-summary",
    )
    per_energy = [post[c]["wall_time_s"] / post[c]["nfev"] for c in COMBO_KEYS]
    reg.add(
        "SecPerEnergy",
        sum(per_energy) / len(per_energy),
        decimals=1,
        source="results/fake_guadalupe_m0-summary, wall time / evaluations",
    )
    # The pre-M0 seed-reuse bug: the code is gone, so these are quoted from the roadmap.
    src = "docs/roadmap.md section 1.3, measured on commit 81bee8b"
    reg.add("SeedBugSpread", 0.22, decimals=2, source=src, quoted=True)
    reg.add("SeedBugGrad", 2, kind="int", source=src, quoted=True)
    reg.add("SeedBugRows", 5, kind="int", source=src, quoted=True)
    # M0 spread check: measured with FakeGuadalupeV2 noise at the exact parameters (40 rows)
    src = "docs/roadmap.md M0 status, with device noise at the exact parameters"
    reg.add("RoadmapSpreadLow", 0.144, decimals=3, source=src, quoted=True)
    reg.add("RoadmapSpreadHigh", 0.060, decimals=3, source=src, quoted=True)


def _m1(reg: Registry) -> None:
    ideal = runs.aggregate("m1_ideal", "ideal_relative_error")
    noisy = runs.by_name("m1_noisy")
    for ansatz in ("hva", "rvb"):
        a = ANSATZ_KEYS[ansatz]
        for method in METHODS:
            m = METHOD_KEYS[method]
            agg = ideal[f"{ansatz}_{method}"]
            src = "results/m1_ideal-summary (mean, sample std over seeds)"
            reg.add(f"MIdeal{a}{m}", agg.mean, kind="pct", decimals=3, source=src)
            reg.add(f"MIdeal{a}{m}Sd", agg.std, kind="pct", decimals=3, source=src)
            row = next(
                r for r in runs.summary("m1_ideal") if r["name"].endswith(f"{ansatz}_{method}-s1")
            )
            per_it, maxiter, shots = _per_iteration(row["path"])
            reg.add(f"MEnergies{a}{m}", per_it, kind="int", source=row["path"])
            if method == "adam_forward":
                reg.add(f"MIters{a}", maxiter, kind="int", source=row["path"])
            reg.add(f"MShots{a}{m}", shots, kind="int", source=row["path"])
            if ansatz == "rvb" and method in {
                k.split("-", 1)[-1].removeprefix("rvb_") for k in noisy
            }:
                nrow = noisy[f"rvb_{method}"]
                reg.add(
                    f"MNoisy{a}{m}",
                    nrow["ideal_relative_error"],
                    kind="pct",
                    decimals=3,
                    source="results/m1_noisy-summary",
                )
                reg.add(
                    f"MNoisyFinal{a}{m}",
                    nrow["final_energy"],
                    decimals=2,
                    source="results/m1_noisy-summary",
                )
    n_seeds = {agg.n for agg in ideal.values()}
    reg.add(
        "MSeeds",
        n_seeds.pop() if len(n_seeds) == 1 else float("nan"),
        kind="int",
        source="results/m1_ideal-summary",
    )
    first_noisy = next(iter(noisy.values()))
    _, noisy_iters, _ = _per_iteration(first_noisy["path"])
    reg.add("MNoisyIters", noisy_iters, kind="int", source=first_noisy["path"])
    # budget per iteration per basis, identical across methods by design
    reg.add(
        "MBudget",
        reg["MEnergiesHvaFwd"].value * reg["MShotsHvaFwd"].value,
        kind="int",
        source="energies per iteration x shots per energy",
    )
    # gradient noise per component in units of the per-energy shot noise sigma
    run = load_run(
        ROOT / next(r for r in runs.summary("m1_ideal") if "rvb_adam_forward" in r["name"])["path"]
    )
    eps = float(run.config.optimizer.fd_eps)
    reg.add("FdEps", eps, decimals=1, source="optimizer.fd_eps of the M1 runs")
    reg.add("SigmaFwd", math.sqrt(2) / eps, decimals=1, source="sqrt(2)/fd_eps")
    reg.add("SigmaCen", math.sqrt(2) / (2 * eps), decimals=1, source="sqrt(2)/(2 fd_eps)")
    reg.add("SigmaPsRvb", math.sqrt(2) / 2, decimals=2, source="sqrt(2)/2, one occurrence")
    reg.add("SigmaPsHva", math.sqrt(2 * 18) / 2, decimals=1, source="sqrt(36)/2, 18 occurrences")
    # the same at an equal shot budget per iteration: sigma = sigma_B * sqrt(m), m energies/iter
    for a in ("Hva", "Rvb"):
        n = reg[f"Params{a}"].value
        k_occ = 18 if a == "Hva" else 1
        m = {key: reg[f"MEnergies{a}{key}"].value for key in ("Fwd", "Cen", "Ps")}
        src = "per-component std x sqrt(energies per iteration), see report section M1"
        reg.add(f"BudgetFwd{a}", math.sqrt(2) / eps * math.sqrt(m["Fwd"]), decimals=1, source=src)
        reg.add(
            f"BudgetCen{a}", math.sqrt(2) / (2 * eps) * math.sqrt(m["Cen"]), decimals=1, source=src
        )
        reg.add(f"BudgetPs{a}", math.sqrt(k_occ / 2) * math.sqrt(m["Ps"]), decimals=1, source=src)
        if m["Fwd"] != n + 1 or m["Cen"] != 2 * n + 1:
            msg = f"unexpected evaluation counts for {a}: {m}"
            raise ValueError(msg)
    spsa_run = load_run(
        ROOT / next(r for r in runs.summary("m1_ideal") if "hva_spsa" in r["name"])["path"]
    )
    reg.add(
        "SpsaLr",
        float(spsa_run.config.optimizer.lr),
        decimals=2,
        source="optimizer.lr of the M1 SPSA runs",
    )
    reg.add(
        "SpsaCalibrated",
        0,
        kind="text",
        source="optimizer.calibrate of the M1 SPSA runs",
        text="calibrated" if spsa_run.config.optimizer.calibrate else "constant",
    )
    reg.add(
        "AdamLr", float(run.config.optimizer.lr), decimals=2, source="optimizer.lr of the M1 runs"
    )


def _stack_label(row: dict[str, object]) -> str:
    return (
        str(row["name"]).split("-", 1)[1].split("_", 1)[1].replace("_", "+").replace("raw", "none")
    )


def _m2_fixed_point(reg: Registry) -> None:
    """Each mitigation stack at fixed parameters (``kagome-vqe benchmark --point``)."""
    for name, p in POINT_KEYS.items():
        rows = runs.evaluation(name)
        src = f"results/{name}/evaluation.json"
        for ansatz in ("hva", "rvb"):
            a = ANSATZ_KEYS[ansatz]
            for stack in STACKS:
                label = "raw" if stack == "none" else stack.replace("+", "_")
                row = rows[f"{ansatz}_{label}"]
                if _stack_label(row) != stack or row["mitigation"] != stack:
                    msg = f"{src}: row {row['name']} is not stack {stack}"
                    raise ValueError(msg)
                k = f"Mt{a}{p}{STACK_KEYS[stack]}"
                reg.add(k, row["mean"], decimals=2, source=src)
                reg.add(f"{k}Bias", row["bias"], decimals=2, source=src)
                reg.add(
                    f"{k}Rel",
                    row["bias"] / abs(row["ideal_energy"]),
                    kind="pct",
                    decimals=2,
                    source=src,
                )
                reg.add(f"{k}Spread", row["spread"], decimals=3, source=src)
                reg.add(
                    f"{k}Below",
                    (row["exact_energy"] - row["mean"])
                    / (row["spread"] / math.sqrt(row["repeats"])),
                    decimals=1,
                    source=f"{src}: (E0 - mean) / (spread / sqrt(repeats))",
                )
            reg.add(f"MtIdeal{a}{p}", rows[f"{ansatz}_raw"]["ideal_energy"], decimals=2, source=src)
        reg.add(f"MtRepeats{p}", rows["hva_raw"]["repeats"], kind="int", source=src)
    rows = runs.evaluation("m2-evaluation-0")
    src = "results/m2-evaluation-0/evaluation.json"
    reg.add("MtShots", rows["hva_raw"]["shots_per_evaluation"], kind="int", source=src)
    reg.add("MtShotsZne", rows["hva_zne"]["shots_per_evaluation"], kind="int", source=src)
    for label, key in (
        ("readout", "Ro"),
        ("readout_zne", "RoZne"),
        ("twin", "Twin"),
        ("readout_twin", "RoTwin"),
    ):
        reg.add(f"MtSetup{key}", rows[f"hva_{label}"]["setup_shots"], kind="int", source=src)
    est = rows["hva_readout_symmetry"]["estimator"]
    e_mean = (sum(est["readout_e0"]) + sum(est["readout_e1"])) / (2 * len(est["readout_e0"]))
    reg.add("MtReadoutErr", e_mean, kind="pct", decimals=1, source=f"{src}: calibration")
    reg.add(
        "MtReadoutErrMax",
        max(est["readout_e0"] + est["readout_e1"]),
        kind="pct",
        decimals=1,
        source=f"{src}: calibration",
    )
    reg.add("MtKept", est["mean_kept_fraction"], kind="pct", decimals=0, source=src)
    for ansatz in ("hva", "rvb"):
        a = ANSATZ_KEYS[ansatz]
        twin = rows[f"{ansatz}_twin"]["estimator"]
        reg.add(f"MtTwinIdeal{a}", twin["ideal_twin_energy"], decimals=2, source=src)
        reg.add(f"MtTwinNoisy{a}", twin["noisy_twin_energy"], decimals=2, source=src)
        reg.add(
            f"MtTwinFactor{a}",
            twin["ideal_twin_energy"] / twin["noisy_twin_energy"],
            decimals=3,
            source=src,
        )


def _m2_vqe(reg: Registry) -> None:
    """Full optimisations with readout correction plus symmetry filtering."""
    vqe = runs.by_name("m2_vqe")
    src = "results/m2_vqe-summary"
    for combo, key in (("hva_spsa", "HvaSpsa"), ("rvb_adam_parameter_shift", "RvbPs")):
        row = vqe[combo]
        reg.add(f"MvFinal{key}", row["final_energy"], decimals=3, source=src)
        reg.add(f"MvStd{key}", row["final_std"], decimals=3, source=src)
        reg.add(f"MvErr{key}", row["final_relative_error"], kind="pct", decimals=2, source=src)
        reg.add(f"MvIdeal{key}", row["ideal_energy"], decimals=3, source=src)
        reg.add(f"MvIdealErr{key}", row["ideal_relative_error"], kind="pct", decimals=2, source=src)
        reg.add(f"MvShots{key}", row["total_shots"], kind="int", source=src)
        reg.add(f"MvNfev{key}", row["nfev"], kind="int", source=src)
        reg.add(
            f"MvBelow{key}",
            (row["exact_energy"] - row["final_energy"]) / row["final_std"],
            decimals=1,
            source=f"{src}: (E0 - final) / final_std",
        )
        run = load_run(ROOT / row["path"])
        reg.add(f"MvIters{key}", int(run.config.optimizer.maxiter), kind="int", source=row["path"])
        reg.add(f"MvShotsPer{key}", int(run.config.backend.shots), kind="int", source=row["path"])


def _m2_exact(reg: Registry, *, recompute: bool) -> None:
    """Infinite-shot bias of each counts-level estimator (``physics.mitigation_exact``)."""
    ex = physics.mitigation_exact(recompute=recompute)
    src = "docsgen.physics.mitigation_exact (cache/mitigation_exact.json)"
    for ansatz in ("hva", "rvb"):
        a = ANSATZ_KEYS[ansatz]
        for theta, p in ((0.0, "Zero"), (0.3, "Pt")):
            exact = ex["exact"][f"{ansatz}_{theta}"]
            for noise, n in (("full", "Full"), ("readout_only", "Ro")):
                key = f"{ansatz}_{theta}_{noise}"
                for mode, m in EXACT_MODE_KEYS.items():
                    e = ex["energies"][key][mode]
                    reg.add(f"Mx{a}{p}{n}{m}", e, decimals=2, source=src)
                    reg.add(
                        f"Mx{a}{p}{n}{m}Rel",
                        (e - exact) / abs(exact),
                        kind="pct",
                        decimals=1,
                        source=src,
                    )
                reg.add(f"Mx{a}{p}{n}Kept", ex["kept"][key], kind="pct", decimals=0, source=src)
    # leading-order over-correction factor: filtered value times prod 1/(1 - e0 - e1) per pair
    k = ex["energies"]["hva_0.0_readout_only"]
    reg.add(
        "MxOverPredicted",
        1 / (1 - 2 * reg["MtReadoutErr"].value) ** 2,
        decimals=3,
        source="1 / (1 - 2 e)^2 with the mean calibrated flip probability e",
    )
    reg.add("MxOverFactor", k["readout_symmetry"] / k["symmetry"], decimals=3, source=src)
    reg.add(
        "MxOverPct",
        reg["MxOverFactor"].value - 1,
        kind="pct",
        decimals=1,
        source=f"{src}: over-correction factor - 1",
    )
    reg.add(
        "MxPredictFull",
        reg["MxOverFactor"].value * ex["energies"]["hva_0.0_full"]["symmetry"],
        decimals=2,
        source=f"{src}: over-correction factor x symmetry-only energy, full noise",
    )


def write_tex(reg: Registry, path: Path = REPORT_TABLES / "numbers.tex") -> Path:
    r"""One ``\providecommand{\V<key>}{...}`` per number, with its source as a comment."""
    lines = ["% Generated by `uv run python -m docsgen`; do not edit.", ""]
    for key, num in sorted(reg.items()):
        flag = " [quoted]" if num.quoted else ""
        lines.append(f"\\providecommand{{\\V{key}}}{{{num.tex()}}}  % {num.source}{flag}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def write_json(reg: Registry, path: Path = SLIDES_DATA / "numbers.json") -> Path:
    """``{key: {value, text, source, quoted}}`` for the deck."""
    data = {
        key: {
            "value": None if math.isnan(n.value) else n.value,
            "text": n.plain(),
            "math": n.math(),
        }
        | {k: v for k, v in asdict(n).items() if k in ("source", "quoted")}
        for key, n in sorted(reg.items())
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    return path
