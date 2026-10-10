"""Numbers recomputed from the package itself, not read from result files.

Cheap facts (spectrum, matchings, energies at zero parameters, transpiled gate counts) are
computed on every run. The two simulator studies (the noise-floor decomposition and the
shot-noise spread) take about a minute, so they are cached as JSON under ``docsgen/cache`` and
recomputed only with ``--recompute``. Seeds are fixed, so a recompute reproduces the cache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.quantum_info import Statevector

from docsgen.paths import CACHE
from kagome_vqe.ansatz import HVAnsatz, RVBAnsatz, ring_bonds
from kagome_vqe.ansatz.unshare import unshare_parameters
from kagome_vqe.backends import resolve_fake_backend
from kagome_vqe.estimators import AerEnergy
from kagome_vqe.hamiltonian import exact_spectrum, heisenberg_hamiltonian
from kagome_vqe.lattice import (
    GUADALUPE_HEAVY_HEX_LOOP,
    KAGOME_HAMILTONIAN_CYCLE,
    Lattice,
    kagome_unit_cell,
    perfect_matchings,
)
from kagome_vqe.transpile import IsaProblem, compile_for_backend

__all__ = [
    "Facts",
    "bound_compilation",
    "facts",
    "hva_landscape",
    "hva_minimum",
    "noise_floor",
    "shot_spread",
]

SEED = 20261010
NOISE_SHOTS = 16384
SPREAD_REPEATS = 40
SPREAD_SHOTS = (1024, 4096)


@dataclass(frozen=True)
class Facts:
    """Exact and structural facts about the 12-site benchmark."""

    spectrum: tuple[float, float, float]
    num_bonds: int
    num_terms: int
    matchings: tuple[tuple[tuple[int, int], ...], ...]
    ring_fraction_cycle: float
    ring_fraction_identity: float
    singlet_bonds_identity: int
    e_zero: dict[str, float]  # "<ansatz>_<order>" -> energy at all-zero parameters
    cx: dict[str, int]  # "<ansatz>[_unshared|_nolayout]" -> two-qubit gates on Guadalupe
    depth: dict[str, int]
    num_parameters: dict[str, int]
    measurement_groups: int
    covering_overlap: float  # |<D_even|D_odd>| of the two singlet coverings


def _covering_state(bonds: list[tuple[int, int]]) -> Statevector:
    from qiskit import QuantumCircuit

    from kagome_vqe.ansatz import singlet_pair

    qc = QuantumCircuit(12)
    for a, b in bonds:
        singlet_pair(qc, a, b)
    return Statevector(qc)


def _energy_at_zero(ansatz: HVAnsatz | RVBAnsatz, lattice: Lattice) -> float:
    circuit = ansatz.build()
    bound = circuit.assign_parameters(np.zeros(circuit.num_parameters))
    value = Statevector(bound).expectation_value(heisenberg_hamiltonian(lattice))
    return float(np.real(value))


@cache
def guadalupe() -> Any:
    """``FakeGuadalupeV2``, the device of the 2022 challenge."""
    return resolve_fake_backend("FakeGuadalupeV2")


@cache
def isa(ansatz: str, *, unshared: bool = False, layout: bool = True) -> IsaProblem:
    """The ansatz compiled for Guadalupe on the heavy-hex loop, as the benchmarks run it."""
    circuit = (HVAnsatz() if ansatz == "hva" else RVBAnsatz()).build()
    if unshared:
        circuit, _ = unshare_parameters(circuit)
    return compile_for_backend(
        circuit,
        heisenberg_hamiltonian(kagome_unit_cell()),
        guadalupe(),
        initial_layout=GUADALUPE_HEAVY_HEX_LOOP if layout else None,
        seed_transpiler=SEED,
    )


@cache
def facts() -> Facts:
    """Compute :class:`Facts` (a few seconds)."""
    cycle = kagome_unit_cell()
    identity = kagome_unit_cell(site_order=None)
    h = heisenberg_hamiltonian(cycle)
    spectrum = tuple(float(x) for x in exact_spectrum(h, k=3))
    odd, even = ring_bonds(identity.num_sites)
    order = range(0, identity.num_sites, 2)
    singlets_identity = sum(identity.is_bond(k, k + 1) for k in order)
    e_zero = {
        f"{name}_{label}": _energy_at_zero(make(), lat)
        for name, make in (("rvb", RVBAnsatz), ("hva", HVAnsatz))
        for label, lat in (("cycle", cycle), ("identity", identity))
    }
    cx: dict[str, int] = {}
    depth: dict[str, int] = {}
    for name in ("hva", "rvb"):
        for suffix, kwargs in (
            ("", {}),
            ("_unshared", {"unshared": True}),
            ("_nolayout", {"layout": False}),
        ):
            problem = isa(name, **kwargs)
            cx[name + suffix] = problem.two_qubit_gates
            depth[name + suffix] = problem.depth
    return Facts(
        spectrum=(spectrum[0], spectrum[1], spectrum[2]),
        num_bonds=cycle.num_bonds,
        num_terms=len(h),
        matchings=tuple(perfect_matchings(cycle)),
        ring_fraction_cycle=cycle.ring_bond_fraction(),
        ring_fraction_identity=identity.ring_bond_fraction(),
        singlet_bonds_identity=singlets_identity,
        e_zero=e_zero,
        cx=cx,
        depth=depth,
        num_parameters={"hva": HVAnsatz().num_parameters, "rvb": RVBAnsatz().num_parameters},
        measurement_groups=isa("hva").measurement_groups,
        covering_overlap=float(abs(_covering_state(even).inner(_covering_state(odd)))),
    )


def _cached(name: str, recompute: bool, compute: Any) -> dict[str, Any]:
    path = CACHE / f"{name}.json"
    if path.exists() and not recompute:
        data: dict[str, Any] = json.loads(path.read_text())
        return data
    data = compute()
    CACHE.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=1) + "\n")
    return data


def noise_floor(*, recompute: bool = False) -> dict[str, Any]:
    """Energy at the exact parameters (all zero) under each noise channel separately.

    Four settings per ansatz on the transpiled Guadalupe circuit, 16384 shots per basis:
    noise-free sampling, readout error only, gate error plus T1/T2 only, and the full model.
    """

    def compute() -> dict[str, Any]:
        from qiskit_aer.noise import NoiseModel

        backend = guadalupe()
        models = {
            "noise_free": None,
            "readout": NoiseModel.from_backend(backend, gate_error=False, thermal_relaxation=False),
            "gate": NoiseModel.from_backend(backend, readout_error=False),
            "full": NoiseModel.from_backend(backend),
        }
        out: dict[str, Any] = {"shots": NOISE_SHOTS, "seed": SEED, "energies": {}, "stds": {}}
        for name in ("hva", "rvb"):
            problem = isa(name)
            zeros = np.zeros((1, problem.circuit.num_parameters))
            for label, model in models.items():
                est = AerEnergy(problem, shots=NOISE_SHOTS, noise_model=model, seed=SEED)
                batch = est.evaluate(zeros)
                out["energies"][f"{name}_{label}"] = float(batch.energies[0])
                out["stds"][f"{name}_{label}"] = float(batch.stds[0])
        return out

    return _cached("noise_floor", recompute, compute)


def shot_spread(*, recompute: bool = False) -> dict[str, Any]:
    """Real spread of repeated sampled energies against the primitive's reported std.

    HVA without gate noise, one energy per call so every repeat gets a fresh simulator seed.
    Two points: the exact parameters (all zero) and the start point ``(pi/4, pi/4)``, the edge
    of the benchmarks' ``x0 = pi/4 * U[0, 1)`` box. At zero the spread vanishes: in each
    measurement basis the two unmatched bonds of every triangle cancel shot by shot, so the
    singlet product is sampled without noise.
    """

    def compute() -> dict[str, Any]:
        problem = compile_for_backend(
            HVAnsatz().build(), heisenberg_hamiltonian(kagome_unit_cell()), None
        )
        n = problem.circuit.num_parameters
        points = {
            "zero": np.zeros((1, n)),
            "start": np.full((1, n), np.pi / 4),
        }
        out: dict[str, Any] = {
            "repeats": SPREAD_REPEATS,
            "seed": SEED,
            "start_point": points["start"][0].tolist(),
            "real": {},
            "reported": {},
            "mean": {},
        }
        for label, x in points.items():
            for shots in SPREAD_SHOTS:
                est = AerEnergy(problem, shots=shots, seed=SEED + shots)
                batches = [est.evaluate(x) for _ in range(SPREAD_REPEATS)]
                energies = np.array([b.energies[0] for b in batches])
                key = f"{label}_{shots}"
                out["real"][key] = float(energies.std(ddof=1))
                out["reported"][key] = float(np.mean([b.stds[0] for b in batches]))
                out["mean"][key] = float(energies.mean())
        return out

    return _cached("shot_spread", recompute, compute)


@cache
def hva_landscape(site_order: str, num: int = 61) -> tuple[np.ndarray, np.ndarray]:
    """Exact HVA energy on a ``num x num`` grid; rows index ``gamma``, columns ``phi``.

    Both angles span ``[-pi, pi]``. The odd layer (``phi``) acts first. With the
    Hamiltonian-cycle order the singlet product is an eigenstate of the even-bond Hamiltonian,
    so on the line ``phi = 0`` the energy is ``-18`` for every ``gamma``: the minimum is a
    whole line, not a point.
    """
    from qiskit.primitives import StatevectorEstimator

    lattice = kagome_unit_cell(
        site_order=None if site_order == "identity" else KAGOME_HAMILTONIAN_CYCLE
    )
    circuit = HVAnsatz().build()  # parameters sort as (gamma00, phi00)
    axis = np.linspace(-np.pi, np.pi, num)
    phi, gamma = np.meshgrid(axis, axis)
    values = np.stack([gamma.ravel(), phi.ravel()], axis=1)
    pub = (circuit, heisenberg_hamiltonian(lattice), values)
    evs = StatevectorEstimator().run([pub]).result()[0].data.evs
    return axis, np.asarray(evs, dtype=float).reshape(num, num)


@cache
def hva_minimum(site_order: str) -> float:
    """Global HVA minimum: best grid point refined by Nelder-Mead on the exact energy."""
    from qiskit.primitives import StatevectorEstimator
    from scipy.optimize import minimize

    axis, energy = hva_landscape(site_order)
    i, j = np.unravel_index(np.argmin(energy), energy.shape)
    lattice = kagome_unit_cell(
        site_order=None if site_order == "identity" else KAGOME_HAMILTONIAN_CYCLE
    )
    circuit, h = HVAnsatz().build(), heisenberg_hamiltonian(lattice)
    estimator = StatevectorEstimator()

    def f(x: np.ndarray) -> float:
        return float(estimator.run([(circuit, h, x)]).result()[0].data.evs)

    res = minimize(
        f,
        np.array([axis[i], axis[j]]),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10},
    )
    return float(min(res.fun, energy.min()))


def bound_compilation(*, recompute: bool = False) -> dict[str, Any]:
    """Parametric versus bound transpilation at a generic point, with full device noise.

    The benchmarks transpile the circuit once with unbound parameters. Level-3 two-qubit
    resynthesis cannot act on parametric gates, so each Heisenberg block keeps 5 CX instead of
    the 3 a general two-qubit unitary needs. Binding the parameters first lets the transpiler
    resynthesise. Compared at ``theta = 0.3`` for every parameter (all-zero parameters would
    compile to the bare singlet circuit), as the retained fraction ``E_noisy / E_exact``.
    """

    def compute() -> dict[str, Any]:
        h = heisenberg_hamiltonian(kagome_unit_cell())
        out: dict[str, Any] = {"theta": 0.3, "shots": NOISE_SHOTS, "seed": SEED, "rows": {}}
        for name, make in (("hva", HVAnsatz), ("rvb", RVBAnsatz)):
            circuit = make().build()
            x = np.full(circuit.num_parameters, 0.3)
            bound = circuit.assign_parameters(x)
            exact = float(np.real(Statevector(bound).expectation_value(h)))
            for label, circ, params in (
                ("parametric", circuit, x[None, :]),
                ("bound", bound, np.zeros((1, 0))),
            ):
                problem = compile_for_backend(
                    circ,
                    h,
                    guadalupe(),
                    initial_layout=GUADALUPE_HEAVY_HEX_LOOP,
                    seed_transpiler=SEED,
                )
                est = AerEnergy(problem, shots=NOISE_SHOTS, noise_backend=guadalupe(), seed=SEED)
                noisy = float(est.evaluate(params).energies[0])
                out["rows"][f"{name}_{label}"] = {
                    "cx": problem.two_qubit_gates,
                    "depth": problem.depth,
                    "exact": exact,
                    "noisy": noisy,
                    "retained": noisy / exact,
                }
        return out

    return _cached("bound_compilation", recompute, compute)
