# neuralmpcx/neural/__init__.py

"""Neural models for integration with :class:`neuralmpcx.wrappers.Mpc`.

Motivation
==========
Many MPC applications benefit from learning-based dynamics (e.g., LSTMs trained in
PyTorch) while still leveraging CasADi for optimization and automatic differentiation.
This subpackage provides first-class, in-package utilities to bridge PyTorch LSTMs to
CasADi and to adapt their I/O to the shapes expected by :class:`neuralmpcx.wrappers.Mpc`.

Overview
========
We expose a single entry point:

- :class:`neuralmpcx.neural.CasadiLSTM`: loads a PyTorch LSTM (optionally with a
  projection head), converts it to CasADi symbolic functions, handles I/O reordering,
  and runs estimate-then-predict forward passes for neural MPC.

Typical usage with neural MPC
=============================
.. code-block:: python

    from neuralmpcx.neural import CasadiLSTM

    model = CasadiLSTM(
        n_context=10, n_inputs=1, hidden_size=128,
        horizon=10, proj_size=1, input_order="y_then_u",
    )
    model.load_state_dict(torch.load("model.pt"))

    mpc.set_neural_dynamics(
        model=model,
        input_order="y_then_u",
        output_bias=b,
        name="F_neural",
    )

API
===
- :class:`CasadiLSTM`
"""

__all__ = [
    "CasadiLSTM",
]

from .casadi_lstm import CasadiLSTM
