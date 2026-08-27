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

"""Deterministic seeds and experiment designs.

Every random draw in the suite descends from one master entropy value, so a
fresh clone reproduces the study. Comparisons are paired: the noise seed depends
on ``(benchmark, replicate)`` alone, and this module draws each design point
once and writes it as an explicit value into every run spec that uses it.

Two properties of the example scripts keep that pairing intact. The plants draw
a full noise vector and then scale it by sigma, without branching on
``sigma == 0``, so the RNG stream advances identically at every noise level, and
``n_context`` reaches only the LSTM warmup, never the plant RNG.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import qmc


#: Master entropy for the whole suite. Changing this invalidates every result.
MASTER_SEED = 20260821

#: Stable integer ids so that ``SeedSequence`` entropy is reproducible across
#: machines and Python versions (``hash()`` is not).
BENCH_ID: Dict[str, int] = {"cstr": 1, "cts": 2}
STREAM_ID: Dict[str, int] = {"noise": 10, "ic": 20, "mismatch": 30}

MAX_SEED = int(np.iinfo(np.uint32).max) + 1


def stream_seed(benchmark: str, stream: str, index: int) -> int:
    """Derive one reproducible 32-bit seed.

    Parameters
    ----------
    benchmark : {"cstr", "cts"}
        Which plant the seed belongs to.
    stream : {"noise", "ic", "mismatch"}
        Which independent random stream. Streams do not interfere, so adding
        initial-condition sampling leaves the noise realizations untouched.
    index : int
        Position within the stream, such as the replicate number.

    Returns
    -------
    int
        A seed in ``[0, 2**32)``.
    """
    if benchmark not in BENCH_ID:
        raise ValueError(f"unknown benchmark {benchmark!r}; expected one of {list(BENCH_ID)}")
    if stream not in STREAM_ID:
        raise ValueError(f"unknown stream {stream!r}; expected one of {list(STREAM_ID)}")
    ss = np.random.SeedSequence(
        [MASTER_SEED, BENCH_ID[benchmark], STREAM_ID[stream], int(index)]
    )
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def noise_seed(benchmark: str, replicate: int) -> int:
    """Seed for the plant's measurement noise, from the replicate alone."""
    return stream_seed(benchmark, "noise", replicate)


def latin_hypercube(
    lower: Sequence[float],
    upper: Sequence[float],
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Draw a scrambled Latin-hypercube sample inside a box.

    Parameters
    ----------
    lower, upper : sequence of float
        Per-dimension box bounds.
    n_samples : int
        Number of points.
    seed : int
        The same seed yields the same design.

    Returns
    -------
    np.ndarray
        Shape ``(n_samples, ndim)``.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape:
        raise ValueError("lower and upper must have the same length")
    sampler = qmc.LatinHypercube(d=len(lower), seed=seed)
    unit = sampler.random(n=int(n_samples))
    return qmc.scale(unit, lower, upper)


def relative_box(
    nominal: Sequence[float], fraction: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Box of ±``fraction`` around ``nominal``, elementwise.

    Returns
    -------
    (lower, upper) : tuple of np.ndarray

    A component whose nominal value is zero collapses to a point. The two-tank
    region therefore centers on the first-setpoint steady state; see
    :func:`cts_steady_state`.
    """
    nominal = np.asarray(nominal, dtype=float)
    return nominal * (1.0 - fraction), nominal * (1.0 + fraction)


def cts_steady_state(h2_target: float, k1: float, k2: float, k3: float,
                     k6: float, k7: float) -> Tuple[float, float]:
    """Equilibrium tank levels of the two-tank plant for a given ``h2``.

    Solving the tank-2 balance ``x2 = k7*x2 + k2*sqrt(x1) - k3*sqrt(x2)`` for the
    upstream level gives the ``h1`` that holds ``h2`` at ``h2_target``:

    ``sqrt(x1) = ((1 - k7) * x2 + k3 * sqrt(x2)) / k2``

    Parameters
    ----------
    h2_target : float
        Desired tank-2 level [m].
    k1, k2, k3, k6, k7 : float
        Plant coefficients (``k1`` and ``k6`` are accepted for symmetry with the
        plant's parameterisation but only tank 2 pins the equilibrium).

    Returns
    -------
    (h1, h2) : tuple of float
        The equilibrium levels [m].
    """
    sqrt_x1 = ((1.0 - k7) * h2_target + k3 * np.sqrt(h2_target)) / k2
    return float(sqrt_x1**2), float(h2_target)


def describe() -> List[str]:
    """Human-readable summary of the seeding scheme, for ``RUNINFO.json``."""
    return [
        f"master_seed={MASTER_SEED}",
        f"bench_ids={BENCH_ID}",
        f"stream_ids={STREAM_ID}",
        "noise seed = f(benchmark, replicate), paired across all swept factors",
    ]
