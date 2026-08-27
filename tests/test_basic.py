# Copyright 2024-2026 Ênio Lopes Júnior
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

"""Smoke tests for a NeuralMPCX install.

Each public entry point the rest of the suite imports is reachable from a
fresh interpreter. A failure here means the package is installed wrong, not
that any behaviour is broken, so nothing below asserts anything numeric.
"""

import pytest


def test_import():
    """The package imports and carries a ``__version__``."""
    import neuralmpcx

    assert hasattr(neuralmpcx, "__version__")


def test_import_nlp():
    """``Nlp`` is re-exported from the top-level package."""
    from neuralmpcx import Nlp

    assert callable(Nlp)


def test_import_solution():
    """``Solution`` is re-exported from the top-level package."""
    from neuralmpcx import Solution

    assert Solution is not None


def test_import_mpc_wrapper():
    """``Mpc`` is reachable from ``neuralmpcx.wrappers``."""
    from neuralmpcx.wrappers import Mpc

    assert callable(Mpc)


def test_import_casadi_lstm():
    """``CasadiLSTM`` is reachable once torch is installed."""
    pytest.importorskip("torch")
    from neuralmpcx.neural import CasadiLSTM

    assert callable(CasadiLSTM)
