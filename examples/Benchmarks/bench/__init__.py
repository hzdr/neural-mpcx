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

"""Parallel benchmark suite for the NeuralMPCX CSTR and two-tank examples.

``run_experiments.py`` expands the JSON tables in ``configs/`` into runs and
executes them with joblib, writing a Parquet store under ``results/``.
``reproduce_all.py`` reads that store and regenerates every figure and table
without simulating anything.

The simulations themselves live in the example scripts (``neural_mpc_cstr.py``,
``neural_mpc_cts.py``, ``nmpc_cstr.py``). This package configures, schedules,
measures and plots them.
"""

__all__ = [
    "config",
    "seeds",
    "metrics",
    "store",
    "adapters",
    "runner",
    "plotstyle",
]
