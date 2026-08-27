# Copyright 2024-2026 Helmholtz-Zentrum Dresden-Rossendorf e.V. (HZDR)
# Authors:
# - Ênio Lopes Júnior
# - Sebastian Felix Reinecke
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark descriptors, run specifications, and JSON experiment expansion.

A run spec is one closed-loop simulation: a flat JSON-serializable dict of
everything that decides its outcome. A run group is a list of specs one worker
executes back to back.

The JSON tables in ``configs/`` declare experiments as cases, each expanding
into run specs::

    {
      "case": "sweep",
      "sweep": {"noise_sigma_pct": [0, 1, 5]},
      "replicates": 20,
      "set": {"use_meas_noise": true}
    }

``sweep`` is a Cartesian product, ``replicates`` multiplies it, ``set`` applies
fixed overrides, and ``sample`` draws design points over initial conditions or
plant parameters.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from . import seeds as _seeds


CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
TABLES_DIR = Path(__file__).resolve().parent.parent / "tables"


@dataclass(frozen=True)
class Benchmark:
    """Static facts about one plant, shared by the runner and the plots.

    Attributes
    ----------
    key : str
        Short identifier used in filenames and in the seed derivation.
    label : str
        Human-readable name for figure titles.
    module : str
        Import path of the example module providing ``simulate(cfg)``.
    nmpc_module : str or None
        Import path of the physics-MPC counterpart. The two-tank has none, so
        its neural-vs-physics comparison is unavailable.
    state_keys, state_labels, state_units : sequence of str
        Per-state naming for storage and axis labels.
    action_keys, action_labels, action_units : sequence of str
    sample_time_s : float
        Control period; the denominator of every real-time-feasibility ratio.
    track_index : int
        Index of the controlled variable that the setpoints refer to.
    setpoint_values : sequence of float
        Tracked-variable setpoint per segment, physical units.
    setpoint_timestamps : sequence of int
        Step index at which each segment begins.
    x_lower, x_upper : sequence of float
        State constraints in physical units, for the violation metrics and the
        trajectory figures.
    x_ranges : sequence of float
        Operating range per state; makes violations dimensionless.
    x_nominal : sequence of float
        Nominal initial condition, physical units.
    stored_normalized : bool
        True when ``simulate`` returns states normalized to [0, 1]. The CSTR
        does; the two-tank returns meters.
    norm_min, norm_max : sequence of float
        Normalization bounds, read only when ``stored_normalized``.
    """

    key: str
    label: str
    module: str
    nmpc_module: str | None
    state_keys: Tuple[str, ...]
    state_labels: Tuple[str, ...]
    state_units: Tuple[str, ...]
    action_keys: Tuple[str, ...]
    action_labels: Tuple[str, ...]
    action_units: Tuple[str, ...]
    sample_time_s: float
    track_index: int
    setpoint_values: Tuple[float, ...]
    setpoint_timestamps: Tuple[int, ...]
    x_lower: Tuple[float, ...]
    x_upper: Tuple[float, ...]
    x_ranges: Tuple[float, ...]
    x_nominal: Tuple[float, ...]
    stored_normalized: bool
    norm_min: Tuple[float, ...]
    norm_max: Tuple[float, ...]

    def to_physical_states(self, x: np.ndarray) -> np.ndarray:
        """Convert a stored state trajectory to physical units."""
        x = np.asarray(x, dtype=float)
        if not self.stored_normalized:
            return x
        lo = np.asarray(self.norm_min, dtype=float)
        hi = np.asarray(self.norm_max, dtype=float)
        return x * (hi - lo) + lo

    def to_physical_actions(self, u: np.ndarray, action_bounds) -> np.ndarray:
        """Convert a stored action trajectory to physical units."""
        u = np.atleast_2d(np.asarray(u, dtype=float))
        if u.shape[0] == 1 and len(self.action_keys) == 1:
            u = u.T
        if not self.stored_normalized:
            return u
        lo = np.asarray([b[0] for b in action_bounds], dtype=float)
        hi = np.asarray([b[1] for b in action_bounds], dtype=float)
        return u * (hi - lo) + lo


#: The CSTR stores everything normalized to [0, 1]; the two-tank stores meters
#: and volts. Both are converted to physical units before any metric is taken.
_CSTR_NORM_MIN = (0.0, 0.0, 0.0, 0.0)
_CSTR_NORM_MAX = (5.1, 5.1, 140.0, 140.0)
CSTR_ACTION_BOUNDS = ((0.0, 100.0), (-8500.0, 0.0))
CTS_ACTION_BOUNDS = ((0.0, 10.0),)

CSTR = Benchmark(
    key="cstr",
    label="CSTR",
    module="neural_mpc_cstr",
    nmpc_module="nmpc_cstr",
    state_keys=("C_A", "C_B", "T_R", "T_K"),
    state_labels=(r"$C_A$", r"$C_B$", r"$T_R$", r"$T_K$"),
    state_units=(r"mol$\cdot$L$^{-1}$", r"mol$\cdot$L$^{-1}$", r"$^\circ$C", r"$^\circ$C"),
    action_keys=("F", "Q_dot"),
    action_labels=(r"$F$", r"$\dot{Q}$"),
    action_units=(r"h$^{-1}$", r"kJ$\cdot$h$^{-1}$"),
    sample_time_s=18.0,
    track_index=1,  # C_B
    setpoint_values=(1.0,),
    setpoint_timestamps=(0,),
    x_lower=(0.1, 0.1, 50.0, 50.0),
    x_upper=(2.0, 2.0, 135.0, 140.0),
    x_ranges=(5.1, 5.1, 140.0, 140.0),
    x_nominal=(0.2, 0.5, 120.0, 120.0),
    stored_normalized=True,
    norm_min=_CSTR_NORM_MIN,
    norm_max=_CSTR_NORM_MAX,
)

CTS = Benchmark(
    key="cts",
    label="Cascaded two-tank",
    module="neural_mpc_cts",
    nmpc_module=None,  # no physics-MPC counterpart exists for this benchmark
    state_keys=("h_1", "h_2"),
    state_labels=(r"$h_1$", r"$h_2$"),
    state_units=("m", "m"),
    action_keys=("u",),
    action_labels=(r"$u$",),
    action_units=("V",),
    sample_time_s=4.0,
    track_index=1,  # h_2
    setpoint_values=(5.0, 8.0, 2.0, 5.0),
    setpoint_timestamps=(0, 200, 400, 600),
    x_lower=(0.0, 0.0),
    x_upper=(10.0, 10.0),
    x_ranges=(10.0, 10.0),
    # Centered on the first-setpoint steady state: a relative box around the
    # empty-tank state (0, 0) collapses to a single point.
    x_nominal=_seeds.cts_steady_state(
        5.0, 0.265885591506958, 0.1621260792016983, 0.15335486829280853,
        1.0295900106430054, 0.9935693740844727,
    ),
    stored_normalized=False,
    norm_min=(0.0, 0.0),
    norm_max=(10.0, 10.0),
)

BENCHMARKS: Dict[str, Benchmark] = {"cstr": CSTR, "cts": CTS}

ACTION_BOUNDS = {"cstr": CSTR_ACTION_BOUNDS, "cts": CTS_ACTION_BOUNDS}


#: Fields a run spec may carry. Expansion rejects anything else in a JSON
#: ``set``/``sweep`` block, so a typo in a config file raises there instead of
#: producing a run that fell back to the default.
SPEC_FIELDS = {
    "experiment", "case", "benchmark", "controller", "replicate",
    "n_context", "hidden_size", "horizon", "n_warmup", "model_name", "shooting",
    "num_iter", "x0",
    "alpha", "beta", "mismatch_factor", "gain_mismatch",
    "use_meas_noise", "noise_sigma_pct", "seed",
    "dist_kind", "dist_magnitude", "dist_onset",
    "use_ekf", "temp_noise_std",
}

#: Per-benchmark nominal values. These reproduce what the example scripts do
#: when run with no arguments, and are the baseline every experiment perturbs.
DEFAULTS: Dict[str, Dict[str, Any]] = {
    "cstr": {
        "controller": "neural",
        "n_context": 10,
        "hidden_size": 16,
        "horizon": 20,
        "n_warmup": 1,
        "num_iter": 60,
        "shooting": "multi",
        "alpha": 1.0,
        "beta": 1.0,
        "use_meas_noise": False,
        "noise_sigma_pct": 0.0,
        "dist_kind": "none",
        "dist_magnitude": 0.0,
        "dist_onset": 0,
        "use_ekf": True,
        "temp_noise_std": 0.5,
        "replicate": 0,
        "model_name": None,
        "x0": None,
    },
    "cts": {
        "controller": "neural",
        "n_context": 10,
        "hidden_size": 128,
        "horizon": 10,
        "n_warmup": 1,
        "num_iter": 1050,
        "shooting": "multi",
        "mismatch_factor": 1.0,
        "gain_mismatch": 1.0,
        "use_meas_noise": False,
        "noise_sigma_pct": 0.0,
        "dist_kind": "none",
        "dist_magnitude": 0.0,
        "dist_onset": 0,
        "replicate": 0,
        "model_name": None,
        "x0": None,
    },
}


def make_run_id(spec: Dict[str, Any]) -> str:
    """Content-addressed identifier for a run.

    A stable digest of the spec, so re-running the suite regenerates the same
    ids and ``--resume`` skips work already in the store. Two specs differing in
    any field get different ids.
    """
    payload = json.dumps(
        {k: v for k, v in sorted(spec.items()) if k != "run_id"},
        sort_keys=True,
        default=str,
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def finalize_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in the derived fields of a spec: the noise seed and the run id.

    The seed is derived from ``(benchmark, replicate)`` only. That is the
    common-random-numbers rule; see :mod:`bench.seeds`.
    """
    spec = dict(spec)
    spec["seed"] = _seeds.noise_seed(spec["benchmark"], int(spec.get("replicate", 0)))
    spec["run_id"] = make_run_id(spec)
    return spec


def _validate_keys(where: str, keys: Iterable[str]) -> None:
    unknown = sorted(set(keys) - SPEC_FIELDS)
    if unknown:
        raise ValueError(
            f"{where}: unknown parameter(s) {unknown}. "
            f"Valid parameters are: {sorted(SPEC_FIELDS)}"
        )


def _sweep_product(sweep: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """Cartesian product of a ``{param: [values]}`` block."""
    if not sweep:
        return [{}]
    names = list(sweep)
    return [
        dict(zip(names, combo))
        for combo in itertools.product(*(sweep[n] for n in names))
    ]


def _sample_points(
    sample: Dict[str, Any], benchmark: Benchmark
) -> List[Dict[str, Any]]:
    """Expand a ``sample`` block into per-run parameter overrides.

    Supported kinds
    ---------------
    ``lhs_x0``
        Latin-hypercube initial conditions in a ±``fraction`` box around the
        benchmark's nominal state. ``fraction`` is one number or a
        ``{benchmark_key: fraction}`` mapping.
    ``lhs_params``
        Latin-hypercube sample of named plant parameters over given ranges.

    Either kind may carry ``seed_index`` (default 0) to move it along its
    stream, which keeps exp3's narrow mismatch sample and its wide envelope from
    sharing structure.
    """
    kind = sample["kind"]
    nominal = np.asarray(benchmark.x_nominal, dtype=float)

    def _fraction(default: float = 0.15) -> float:
        frac = sample.get("fraction", default)
        if isinstance(frac, dict):
            frac = frac[benchmark.key]
        return float(frac)

    if kind == "lhs_x0":
        frac = _fraction()
        lo, hi = _seeds.relative_box(nominal, frac)
        pts = _seeds.latin_hypercube(
            lo, hi, int(sample["n"]),
            seed=_seeds.stream_seed(
                benchmark.key, "ic", int(sample.get("seed_index", 0))
            ),
        )
        return [{"x0": [float(v) for v in p]} for p in pts]

    if kind == "lhs_params":
        params = sample["params"]
        _validate_keys("sample.params", params)
        names = list(params)
        lo = [params[n][0] for n in names]
        hi = [params[n][1] for n in names]
        pts = _seeds.latin_hypercube(
            lo, hi, int(sample["n"]),
            seed=_seeds.stream_seed(
                benchmark.key, "mismatch", int(sample.get("seed_index", 0))
            ),
        )
        return [dict(zip(names, (float(v) for v in p))) for p in pts]

    raise ValueError(f"unknown sample kind {kind!r}")


@dataclass
class Experiment:
    """One JSON experiment table, parsed."""

    id: str
    name: str
    description: str = ""
    benchmarks: List[str] = field(default_factory=lambda: ["cstr", "cts"])
    timing_critical: bool = False
    group_by: List[str] = field(default_factory=list)
    cases: List[Dict[str, Any]] = field(default_factory=list)
    per_benchmark: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    set: Dict[str, Any] = field(default_factory=dict)


def load_experiment(path: Path) -> Experiment:
    """Parse one experiment JSON file."""
    raw = json.loads(Path(path).read_text())
    known = {"id", "name", "description", "benchmarks", "timing_critical",
             "group_by", "cases", "per_benchmark", "set"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{path.name}: unknown top-level key(s) {unknown}")
    return Experiment(
        id=raw["id"],
        name=raw.get("name", raw["id"]),
        description=raw.get("description", ""),
        benchmarks=raw.get("benchmarks", ["cstr", "cts"]),
        timing_critical=bool(raw.get("timing_critical", False)),
        group_by=raw.get("group_by", []),
        cases=raw.get("cases", []),
        per_benchmark=raw.get("per_benchmark", {}),
        set=raw.get("set", {}),
    )


def load_all_experiments(config_dir: Path | None = None) -> List[Experiment]:
    """Load every ``exp*.json`` table, sorted by id."""
    config_dir = Path(config_dir or CONFIG_DIR)
    files = sorted(p for p in config_dir.glob("exp*.json"))
    return [load_experiment(p) for p in files]


def expand(exp: Experiment) -> List[List[Dict[str, Any]]]:
    """Expand an experiment into run groups.

    Returns
    -------
    list of list of dict
        Each inner list is one joblib task. ``group_by`` names fields whose
        matching specs share a worker, so a comparison needing both members in
        one process can avoid a cross-run join. No experiment sets it today, so
        each spec forms its own group.
    """
    specs: List[Dict[str, Any]] = []

    for bench_key in exp.benchmarks:
        if bench_key not in BENCHMARKS:
            raise ValueError(f"{exp.id}: unknown benchmark {bench_key!r}")
        benchmark = BENCHMARKS[bench_key]
        base = dict(DEFAULTS[bench_key])
        _validate_keys(f"{exp.id}.set", exp.set)
        base.update(exp.set)
        per_bench = exp.per_benchmark.get(bench_key, {})
        _validate_keys(f"{exp.id}.per_benchmark.{bench_key}", per_bench)
        base.update(per_bench)

        for case in exp.cases:
            case_name = case.get("case", "default")
            if "benchmarks" in case and bench_key not in case["benchmarks"]:
                continue

            case_set = case.get("set", {})
            _validate_keys(f"{exp.id}.{case_name}.set", case_set)
            case_per_bench = case.get("per_benchmark", {}).get(bench_key, {})
            _validate_keys(
                f"{exp.id}.{case_name}.per_benchmark.{bench_key}", case_per_bench
            )

            sweep = case.get("sweep", {})
            _validate_keys(f"{exp.id}.{case_name}.sweep", sweep)
            sweep_rows = _sweep_product(sweep)

            if "sample" in case:
                sample_rows = _sample_points(case["sample"], benchmark)
            else:
                sample_rows = [{}]

            n_reps = int(case.get("replicates", 1))

            for sweep_row, sample_row, rep in itertools.product(
                sweep_rows, sample_rows, range(n_reps)
            ):
                spec = dict(base)
                spec.update(case_set)
                spec.update(case_per_bench)
                spec.update(sweep_row)
                spec.update(sample_row)
                spec["replicate"] = rep
                spec["experiment"] = exp.id
                spec["benchmark"] = bench_key
                spec["case"] = case_name
                # NMPC has no LSTM, so these fields carry no meaning there.
                # Normalizing them keeps the run ids stable.
                if spec.get("controller") == "nmpc":
                    spec["n_context"] = 1
                    spec["hidden_size"] = -1
                    spec["model_name"] = None
                specs.append(finalize_spec(spec))

    # De-duplicate: two cases can define the same run, such as a mismatch case
    # sampled at nominal parameters and the nominal reference. One copy is
    # simulated and both cases refer to it.
    unique: Dict[str, Dict[str, Any]] = {}
    for spec in specs:
        unique.setdefault(spec["run_id"], spec)

    if not exp.group_by:
        return [[spec] for spec in unique.values()]

    groups: Dict[Tuple, List[Dict[str, Any]]] = {}
    for spec in unique.values():
        key = tuple(spec.get(f) for f in exp.group_by)
        groups.setdefault(key, []).append(spec)
    return list(groups.values())


__all__ = [
    "Benchmark", "BENCHMARKS", "CSTR", "CTS", "ACTION_BOUNDS",
    "DEFAULTS", "SPEC_FIELDS", "Experiment",
    "load_experiment", "load_all_experiments", "expand",
    "make_run_id", "finalize_spec",
    "CONFIG_DIR", "RESULTS_DIR", "FIGURES_DIR", "TABLES_DIR",
]
