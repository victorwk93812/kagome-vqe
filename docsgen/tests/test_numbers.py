"""Every number the report and the deck show is asserted here.

Run with ``uv run pytest docsgen/tests``. Tests that read ``results/`` skip when the symlink is
missing (a fresh clone); the physics tests always run.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from docsgen import numbers, physics, traces
from docsgen.paths import RESULTS, ROOT, SLIDES_DATA
from kagome_vqe.ansatz import ring_bonds
from kagome_vqe.lattice import KAGOME_HAMILTONIAN_CYCLE

needs_results = pytest.mark.skipif(not RESULTS.exists(), reason="results/ is not available")


# -- physics -----------------------------------------------------------------------------------


def test_spectrum_two_fold_ground_state() -> None:
    f = physics.facts()
    assert f.spectrum[0] == pytest.approx(-18.0, abs=1e-9)
    assert f.spectrum[1] == pytest.approx(-18.0, abs=1e-9)
    assert f.spectrum[2] == pytest.approx(-16.9613, abs=1e-4)
    assert len(f.matchings) == 2
    assert f.num_terms == 3 * f.num_bonds == 54


def test_matchings_are_the_two_ring_layers() -> None:
    """Even ring bonds = one dimer covering, odd ring bonds = the other (cycle order)."""
    odd, even = ring_bonds(12)
    order = KAGOME_HAMILTONIAN_CYCLE

    def sites(bonds: list[tuple[int, int]]) -> set[frozenset[int]]:
        return {frozenset((order[a], order[b])) for a, b in bonds}

    matchings = {frozenset(frozenset(b) for b in m) for m in physics.facts().matchings}
    assert matchings == {frozenset(sites(even)), frozenset(sites(odd))}


def test_covering_overlap_is_two_to_the_loops_minus_half_n() -> None:
    """The two coverings form one 12-loop: |<D1|D2>| = 2^(1 - 12/2) = 1/32."""
    assert physics.facts().covering_overlap == pytest.approx(1 / 32, abs=1e-12)


def test_ring_is_union_of_coverings() -> None:
    order = KAGOME_HAMILTONIAN_CYCLE
    ring = {frozenset((order[k], order[(k + 1) % 12])) for k in range(12)}
    union = {frozenset(b) for m in physics.facts().matchings for b in m}
    assert ring == union


def test_site_order_facts() -> None:
    f = physics.facts()
    assert f.ring_fraction_cycle == 1.0
    assert f.ring_fraction_identity == 0.5
    assert f.singlet_bonds_identity == 3
    for ansatz in ("rvb", "hva"):
        assert f.e_zero[f"{ansatz}_cycle"] == pytest.approx(-18.0, abs=1e-9)
        assert f.e_zero[f"{ansatz}_identity"] == pytest.approx(-9.0, abs=1e-9)


def test_hva_identity_minimum_is_far_from_ground_state() -> None:
    assert physics.hva_minimum("identity") == pytest.approx(-10.71, abs=0.01)
    assert physics.hva_minimum("cycle") == pytest.approx(-18.0, abs=1e-6)


def test_cycle_landscape_minimum_is_a_line() -> None:
    """With the cycle order, phi = 0 gives -18 for every gamma (singlets are H_even eigenstates)."""
    axis, energy = physics.hva_landscape("cycle")
    column = energy[:, int(np.argmin(np.abs(axis)))]
    assert np.allclose(column, -18.0, atol=1e-9)
    assert np.ptp(energy, axis=0).max() > 1.0  # away from phi = 0, gamma matters


def test_transpiled_cost_and_unsharing() -> None:
    f = physics.facts()
    assert f.cx["hva"] == f.cx["rvb"] == 66
    assert f.cx["hva_unshared"] == f.cx["hva"]
    assert f.cx["rvb_unshared"] == f.cx["rvb"]
    assert f.measurement_groups == 3


def test_noise_floor_is_nearly_multiplicative() -> None:
    nf = physics.noise_floor()["energies"]
    for ansatz in ("hva", "rvb"):
        ideal = nf[f"{ansatz}_noise_free"]
        assert ideal == pytest.approx(-18.0, abs=1e-9)  # zero sampling noise at the optimum
        product = ideal * (nf[f"{ansatz}_readout"] / ideal) * (nf[f"{ansatz}_gate"] / ideal)
        assert product == pytest.approx(nf[f"{ansatz}_full"], abs=0.2)
        assert nf[f"{ansatz}_full"] > nf[f"{ansatz}_gate"] > nf[f"{ansatz}_readout"] > ideal


def test_shot_spread() -> None:
    sp = physics.shot_spread()
    assert sp["real"]["zero_1024"] == 0.0
    assert sp["real"]["zero_4096"] == 0.0
    # 1/sqrt(shots): quadrupling shots halves the spread, within sampling error of 40 repeats
    assert sp["real"]["start_1024"] / sp["real"]["start_4096"] == pytest.approx(2.0, rel=0.35)
    assert sp["reported"]["start_1024"] / sp["real"]["start_1024"] > 5


# -- traces ------------------------------------------------------------------------------------


def test_traces_validate_and_end_at_ground_state() -> None:
    ring = traces.ring_build()
    traces.validate(ring)
    energies = [s["readouts"]["energy"] for s in ring["steps"]]
    assert energies[0] == "+18.0"
    assert energies[-1] == "−18.0"
    order = traces.site_order()
    traces.validate(order)
    assert [s["readouts"]["energy"] for s in order["steps"]] == ["−9.0", "−18.0"]


def test_written_traces_match_generator() -> None:
    for name, make in traces.ALL.items():
        path = SLIDES_DATA / f"{name}.json"
        assert json.loads(path.read_text()) == json.loads(json.dumps(make(), ensure_ascii=False)), (
            name
        )


# -- registry and documents --------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry() -> numbers.Registry:
    if not RESULTS.exists():
        pytest.skip("results/ is not available")
    return numbers.collect()


@needs_results
def test_quoted_numbers_are_the_known_ones(registry: numbers.Registry) -> None:
    quoted = {k for k, v in registry.items() if v.quoted}
    assert quoted == {"SeedBugSpread", "SeedBugGrad", "SeedBugRows"}


@needs_results
def test_every_macro_in_the_report_exists(registry: numbers.Registry) -> None:
    used: set[str] = set()
    for tex in (ROOT / "report").rglob("*.tex"):
        if tex.name == "numbers.tex":
            continue
        used |= set(re.findall(r"\\V([A-Z][A-Za-z]*)", tex.read_text()))
    missing = sorted(used - set(registry))
    assert not missing, f"report uses undefined numbers: {missing}"


@needs_results
def test_every_number_in_the_deck_exists(registry: numbers.Registry) -> None:
    html = "".join(p.read_text() for p in (ROOT / "slides").glob("*.html"))
    used = set(re.findall(r"\{\{([A-Za-z]+)(?::m)?\}\}", html))
    assert used, "the deck quotes no numbers"
    missing = sorted(used - set(registry))
    assert not missing, f"deck uses undefined numbers: {missing}"


@needs_results
def test_numbers_files_are_current(registry: numbers.Registry) -> None:
    data = json.loads((SLIDES_DATA / "numbers.json").read_text())
    assert {k: v["text"] for k, v in data.items()} == {k: n.plain() for k, n in registry.items()}


@needs_results
def test_m1_means_match_readme(registry: numbers.Registry) -> None:
    """The README's gradient table quotes the same means (it uses population std, ddof=0)."""
    readme = (ROOT / "README.md").read_text()
    pattern = re.compile(r"\| (HVA|RVB) \| ([^|]+?) \| (\d+) \| ([\d.]+) % ± ([\d.]+) \|")
    labels = {
        "ADAM, forward differences (the paper)": "Fwd",
        "ADAM, central differences": "Cen",
        "ADAM, SPSA gradient (hybrid)": "Hyb",
        "ADAM, parameter shift": "Ps",
        "plain SPSA": "Spsa",
    }
    rows = pattern.findall(readme)
    assert len(rows) == 10
    for ansatz, method, energies, mean, _ in rows:
        key = f"{ansatz.capitalize()}{labels[method]}"
        assert registry[f"MEnergies{key}"].value == pytest.approx(float(energies))
        assert 100 * registry[f"MIdeal{key}"].value == pytest.approx(float(mean), abs=6e-4)


@needs_results
def test_noisy_baseline(registry: numbers.Registry) -> None:
    assert registry["PostFinalMin"].value == pytest.approx(-12.87, abs=0.01)
    assert registry["PostFinalMax"].value == pytest.approx(-12.67, abs=0.01)
    assert registry["PreFinalRvbAdam"].value == pytest.approx(-10.76, abs=0.01)
    assert registry["MBudget"].value == 24576
