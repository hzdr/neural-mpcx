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

"""Tests for the parallel benchmark suite in ``examples/Benchmarks``.

These guard the two properties that are easy to break silently and expensive to
discover late: the common-random-numbers contract that makes every comparison
paired, and the detection of failed solves.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "examples" / "Benchmarks"
for path in (str(ROOT), str(BENCH_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from bench import config as bconfig  # noqa: E402
from bench import metrics as bmetrics  # noqa: E402
from bench import seeds as bseeds  # noqa: E402


# --------------------------------------------------------------------------
# Common random numbers
# --------------------------------------------------------------------------

def test_noise_seed_depends_only_on_benchmark_and_replicate():
    """The whole paired design rests on this: the seed ignores swept factors."""
    base = dict(bconfig.DEFAULTS["cstr"])
    base.update({"experiment": "t", "case": "t", "benchmark": "cstr", "replicate": 7})

    variants = [
        {"n_context": 5, "noise_sigma_pct": 0.0, "hidden_size": 16, "horizon": 20},
        {"n_context": 80, "noise_sigma_pct": 5.0, "hidden_size": 128, "horizon": 40},
    ]
    seeds = set()
    for extra in variants:
        spec = dict(base)
        spec.update(extra)
        seeds.add(bconfig.finalize_spec(spec)["seed"])
    assert len(seeds) == 1, "the noise seed must not depend on any swept factor"


def test_noise_seed_differs_across_replicates_and_benchmarks():
    assert bseeds.noise_seed("cstr", 0) != bseeds.noise_seed("cstr", 1)
    assert bseeds.noise_seed("cstr", 3) != bseeds.noise_seed("cts", 3)


def test_seeds_are_reproducible():
    assert bseeds.stream_seed("cts", "ic", 4) == bseeds.stream_seed("cts", "ic", 4)
    a = bseeds.latin_hypercube([0, 0], [1, 1], 16, seed=99)
    b = bseeds.latin_hypercube([0, 0], [1, 1], 16, seed=99)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("benchmark", ["cstr", "cts"])
def test_identical_noise_realisation_across_sigma(benchmark):
    """Replicate r sees the same noise samples at every sigma.

    Drawn from the plant, where the stream lives. If an environment branches on
    sigma or consumes the generator for anything else, the paired design becomes
    an unpaired one with no error to show for it.
    """
    module = _load_plant_module(benchmark)
    seed = bseeds.noise_seed(benchmark, 5)

    def draw(sigma):
        cfg = module.RunConfig(use_meas_noise=True, noise_sigma_pct=sigma, seed=seed)
        env = _make_env(module, benchmark, cfg)
        env.reset(seed=seed)
        return np.array([env.np_random.standard_normal((env.nx, 1)).ravel()
                         for _ in range(12)])

    assert np.array_equal(draw(0.0), draw(5.0)), (
        "the RNG stream must advance identically at every sigma; the plants "
        "must always draw and then scale, never branch on sigma == 0"
    )


def _load_plant_module(benchmark):
    example_dir = ROOT / (
        "examples/CSTR" if benchmark == "cstr"
        else "examples/Cascaded_Two_Tank_System"
    )
    if str(example_dir) not in sys.path:
        sys.path.insert(0, str(example_dir))
    import importlib

    return importlib.import_module(
        "neural_mpc_cstr" if benchmark == "cstr" else "neural_mpc_cts"
    )


def _make_env(module, benchmark, cfg):
    return module.CSTRSystem(cfg) if benchmark == "cstr" else module.NtiSystem(cfg)


def test_zero_noise_runs_are_bit_identical():
    """Determinism check: nothing unseeded leaks into a noise-free run."""
    module = _load_plant_module("cstr")
    cfg = module.RunConfig(num_iter=4)
    a = module.simulate(cfg)
    b = module.simulate(cfg)
    assert np.array_equal(a["X"], b["X"])
    assert np.array_equal(a["U"], b["U"])


# --------------------------------------------------------------------------
# Failure detection
# --------------------------------------------------------------------------

def test_failed_solves_are_detected_and_predictions_nulled():
    """A failed solve must be counted, and must not leave a stale prediction.

    Under the default ``last-successful`` warm start, ``_last_solution`` is not
    replaced on failure, so harvesting a prediction from it would silently
    repeat the previous successful solve. Contradictory hard bounds make every
    solve infeasible, which is the cheapest deterministic way to provoke that.
    """
    import copy

    module = _load_plant_module("cstr")
    pars = copy.deepcopy(module.NeuralMpc.pars_init)
    # A hard-constrained state (index 0) with a lower bound above its upper
    # bound: infeasible at every step, by construction.
    pars["x_lb"] = np.asarray([0.9, 0.02, 0.35, 0.35], dtype=float)
    pars["x_ub"] = np.asarray([0.1, 0.39, 0.96, 1.0], dtype=float)

    cfg = module.RunConfig(num_iter=3, pars_init=pars)
    result = module.simulate(cfg)

    assert result["n_failed_solves"] > 0, "an infeasible problem must be counted"
    failed = ~result["solve_ok"]
    preds = np.atleast_2d(result["X_pred"])
    assert np.isnan(preds[failed]).all(), (
        "predictions from a failed step must be NaN, not a stale repeat"
    )

    spec = bconfig.finalize_spec(
        {**bconfig.DEFAULTS["cstr"], "experiment": "t", "case": "t",
         "benchmark": "cstr", "num_iter": 3}
    )
    from bench import adapters as badapters

    row = badapters.summarize(spec, result)
    assert row["completed"] is False
    assert row["n_failed_solves"] == result["n_failed_solves"]


# --------------------------------------------------------------------------
# Configuration expansion
# --------------------------------------------------------------------------

def test_run_id_is_stable_and_specification_sensitive():
    base = {**bconfig.DEFAULTS["cts"], "experiment": "t", "case": "t",
            "benchmark": "cts"}
    first = bconfig.finalize_spec(dict(base))["run_id"]
    again = bconfig.finalize_spec(dict(base))["run_id"]
    assert first == again
    changed = bconfig.finalize_spec({**base, "horizon": 40})["run_id"]
    assert first != changed


def test_unknown_parameter_in_a_config_is_rejected():
    """A typo in a JSON table must fail loudly, not silently use the default."""
    exp = bconfig.Experiment(
        id="bad", name="bad", benchmarks=["cstr"],
        cases=[{"case": "c", "sweep": {"nocontext": [1, 2]}}],
    )
    with pytest.raises(ValueError, match="unknown parameter"):
        bconfig.expand(exp)


def test_every_shipped_config_expands():
    """Each committed experiment table must expand without error."""
    for exp in bconfig.load_all_experiments():
        groups = bconfig.expand(exp)
        assert groups, f"{exp.id} expanded to nothing"
        for group in groups:
            for spec in group:
                assert spec["benchmark"] in bconfig.BENCHMARKS
                assert "run_id" in spec and "seed" in spec


def test_checkpoints_named_by_the_configs_exist():
    """Guard against a hidden-size sweep referring to a missing checkpoint."""
    for exp in bconfig.load_all_experiments():
        for group in bconfig.expand(exp):
            for spec in group:
                if spec.get("controller") != "neural":
                    continue
                bench = bconfig.BENCHMARKS[spec["benchmark"]]
                module = _load_plant_module(spec["benchmark"])
                cfg_cls = module.RunConfig
                accepted = set(cfg_cls.__dataclass_fields__)
                cfg = cfg_cls(**{k: v for k, v in spec.items()
                                 if k in accepted and v is not None})
                sub = ("CSTR" if bench.key == "cstr"
                       else "Cascaded_Two_Tank_System")
                path = (ROOT / "examples" / sub / "models"
                        / f"{cfg.resolved_model_name()}.pt")
                assert path.exists(), f"{exp.id}: missing checkpoint {path}"


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_failed_runs_are_ranked_worst_not_dropped():
    """A non-finite metric must still count toward the sample and the median."""
    stats = bmetrics.median_iqr([1.0, 2.0, 3.0, np.inf])
    assert stats["n"] == 4
    assert stats["median"] > 2.0


def test_recovery_time_is_infinite_when_never_recovered():
    tracked = np.full(20, 10.0)
    setpoint = np.full(20, 1.0)
    assert bmetrics.recovery_steps(tracked, setpoint, onset=0) == float("inf")


def test_reached_band_final_versus_all_segments():
    """The two readings must differ when an intermediate segment misses."""
    tracked = np.concatenate([np.full(10, 9.0), np.full(10, 5.0)])
    setpoints, timestamps = [2.0, 5.0], [0, 10]
    assert bmetrics.reached_band(tracked, setpoints, timestamps) is True
    assert bmetrics.reached_band(
        tracked, setpoints, timestamps, all_segments=True
    ) is False


def test_violations_are_dimensionless_and_counted():
    x = np.array([[0.0, 5.0], [12.0, 5.0]])
    out = bmetrics.violations(x, lower=[0.0, 0.0], upper=[10.0, 10.0],
                              ranges=[10.0, 10.0])
    assert out["n_violations"] == 1
    assert out["violation_rate"] == 0.5
    assert out["worst_violation"] == pytest.approx(0.2)


def test_total_variation_counts_every_move():
    assert bmetrics.total_variation(np.array([0.0, 1.0, 0.0, 2.0])) == 4.0
