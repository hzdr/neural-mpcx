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

"""PyTorch LSTM wrapper for Neural MPC with context-based state estimation."""

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """LSTM neural network with projection layer for system identification.

    This LSTM implementation supports context-based state estimation where the
    initial hidden state is estimated using a window of past observations before
    making predictions, following the approach in "Learning in MPC: Learning
    Initial State Estimation for Recurrent Neural Network Dynamics".
    """

    def __init__(
        self,
        n_context: int,
        n_inputs: int,
        hidden_size: int,
        batch_size: int,
        sequence_length: int,
        n_outputs: int,
        is_estimator: bool = True,
        dropout_rate: float = 0.2,
    ):

        super(LSTM, self).__init__()
        self.n_outputs = n_outputs
        self.n_context = n_context  # Number of steps for context
        self.n_inputs = n_inputs  # Input size (num features)
        self.sequence_length = sequence_length
        self.is_estimator = is_estimator
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        # Define the LSTM model with projection
        self.model = nn.LSTM(
            input_size=n_inputs,
            hidden_size=self.hidden_size,
            proj_size=self.n_outputs,
            num_layers=1,
            batch_first=True,
        )

        # Define independent dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

        self.hn = None
        self.cn = None

        # Apply custom initialization to LSTM parameters
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize LSTM weights with small Gaussian noise and biases to zero."""
        for name, param in self.model.named_parameters():
            if "weight" in name:
                # Initialize weights with Gaussian distribution
                nn.init.normal_(param, mean=0.0, std=1e-4)
            elif "bias" in name:
                # Initialize biases to zero
                nn.init.constant_(param, 0)

    def forward(self, u_train):
        """Forward pass through LSTM with context-based state estimation.

        Parameters
        ----------
        u_train : torch.Tensor
            Input features, shape (batch_size, sequence_length, input_size) or
            (batch_size * sequence_length, input_size).

        Returns
        -------
        y_sim : torch.Tensor
            Predicted outputs, shape (batch_size, sequence_length, output_size).
        """
        if (len(u_train.shape)) == 2:
            batch_size_sequence_length = u_train.shape[0]

            if batch_size_sequence_length % self.sequence_length == 0:
                batch_size = batch_size_sequence_length // self.sequence_length
            else:
                raise ValueError(
                    "The total size is not divisible by the sequence length."
                )

            u_train = u_train.view(batch_size, self.sequence_length, -1)

        if self.is_estimator:
            y1 = self.estimate_state(
                u_train[:, :, : self.n_inputs],
                u_train[:, :, self.n_inputs :],
                self.n_context,
            )
            y2 = self.predict_state(u_train[:, :, : self.n_inputs], self.n_context)

            y_sim = torch.cat((y1, y2), dim=1)
        else:
            state = (self.hn, self.cn)
            y_sim, _ = self.model(u_train, state)
            y_sim = self.dropout(y_sim)

        return y_sim

    def estimate_state(self, u_train, y_train, nstep):
        """Estimate LSTM hidden state using measured outputs over context window.

        Parameters
        ----------
        u_train : torch.Tensor
            Input sequence for state estimation.
        y_train : torch.Tensor
            Measured output sequence used to warm up hidden states.
        nstep : int
            Number of context steps for state estimation.

        Returns
        -------
        y_sim : torch.Tensor
            Estimated outputs over context window.
        """
        if len(u_train.shape) == 2:
            u_train = u_train.view(self.batch_size, u_train.shape[0], self.n_inputs)
        if len(y_train.shape) == 2:
            y_train = y_train.view(self.batch_size, y_train.shape[0], self.n_outputs)

        y_est = []
        device = u_train.device
        hn = torch.zeros(
            1, u_train.size(0), self.n_outputs, requires_grad=True, device=device
        )
        cn = torch.zeros(
            1, u_train.size(0), self.hidden_size, requires_grad=True, device=device
        )

        for i in range(nstep):
            h0 = y_train[:, i, :].contiguous().view(hn.shape)
            c0 = cn.contiguous()
            inp_i = u_train[:, i, :].unsqueeze(1).contiguous()
            out, (hn, cn) = self.model(inp_i, (h0, c0))
            hn = hn.contiguous()
            cn = cn.contiguous()
            y_est.append(self.dropout(out))

        y_sim = torch.cat(y_est, dim=1)
        self.hn, self.cn = (hn, cn)
        return y_sim

    def predict_state(self, u_train, nstep):
        """Predict future outputs using stored hidden state from estimation phase.

        Parameters
        ----------
        u_train : torch.Tensor
            Input sequence for prediction.
        nstep : int
            Number of context steps (skipped from the beginning).

        Returns
        -------
        y_sim : torch.Tensor
            Predicted outputs for future horizon.
        """
        if len(u_train.shape) == 2:
            batch_size_sequence_length = u_train.shape[0]
            batch_size = batch_size_sequence_length // self.sequence_length
            u_train = u_train.view(batch_size, self.sequence_length, self.n_inputs)
        state = (self.hn.contiguous(), self.cn.contiguous())
        y_sim, _ = self.model(u_train[:, nstep:, :], state)
        return self.dropout(y_sim)

    def get_model(self):
        """Return the underlying LSTM model.

        Returns
        -------
        nn.LSTM
            The PyTorch LSTM module.
        """
        return self.model
