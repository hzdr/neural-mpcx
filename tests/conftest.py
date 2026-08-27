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

"""Shared fixtures for NeuralMPCX test suite."""

from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def ipopt_opts():
    """IPOPT solver options with suppressed output for fast testing."""
    return {
        "print_time": False,
        "ipopt": {
            "print_level": 0,
            "sb": "yes",
            "max_iter": 300,
            "tol": 1e-4,
            "acceptable_tol": 1e-2,
        },
    }


@pytest.fixture(scope="session")
def cts_model_path():
    """Path to smallest CTS LSTM model (hidden_size=8)."""
    p = PROJECT_ROOT / "examples" / "Cascaded_Two_Tank_System" / "models" / "cts-lstm-batched-8.pt"
    if not p.exists():
        pytest.skip(f"CTS model not found at {p}")
    return p


@pytest.fixture(scope="session")
def cstr_model_path():
    """Path to smallest CSTR LSTM model (hidden_size=8)."""
    p = PROJECT_ROOT / "examples" / "CSTR" / "models" / "cstr-lstm-batched-8.pt"
    if not p.exists():
        pytest.skip(f"CSTR model not found at {p}")
    return p


@pytest.fixture(scope="module")
def cts_state_dict(cts_model_path):
    """Loaded CTS model state dict (requires torch)."""
    torch = pytest.importorskip("torch")
    return torch.load(str(cts_model_path), map_location="cpu")


@pytest.fixture(scope="module")
def cstr_state_dict(cstr_model_path):
    """Loaded CSTR model state dict (requires torch)."""
    torch = pytest.importorskip("torch")
    return torch.load(str(cstr_model_path), map_location="cpu")


@pytest.fixture
def make_lstm_state_dict():
    """Factory building a synthetic single/multi-layer LSTM state_dict.

    Returns plain NumPy arrays keyed like a PyTorch LSTM (``model.weight_ih_l*``
    etc.), so a ``CasadiLSTM`` of arbitrary input width can be constructed
    without a pre-trained ``.pt`` file. Used for disturbance tests where the core
    input width is ``n_inputs + n_disturbances``.
    """

    def _make(n_inputs, hidden_size, proj_size=0, num_layers=1, seed=0):
        rng = np.random.default_rng(seed)
        h_out = proj_size if proj_size > 0 else hidden_size
        sd = {}
        for l in range(num_layers):
            in_dim = n_inputs if l == 0 else h_out
            sd[f"model.weight_ih_l{l}"] = (
                rng.standard_normal((4 * hidden_size, in_dim)) * 0.3
            )
            sd[f"model.weight_hh_l{l}"] = (
                rng.standard_normal((4 * hidden_size, h_out)) * 0.3
            )
            sd[f"model.bias_ih_l{l}"] = rng.standard_normal((4 * hidden_size,)) * 0.1
            sd[f"model.bias_hh_l{l}"] = rng.standard_normal((4 * hidden_size,)) * 0.1
            if proj_size > 0:
                sd[f"model.weight_hr_l{l}"] = (
                    rng.standard_normal((proj_size, hidden_size)) * 0.3
                )
        return sd

    return _make
