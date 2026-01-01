import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class LSTMCell(nn.Module):

    """
    ===== LSTM Cell =====

    Equations:
    f_t = sigmoid(W_f @ [h_{t-1], x_t}] + b_f) -> Forget gate
    i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i) -> Input gate
    g_t = tanh(W_g @ [h_{t-1}, x_t] + b_g) -> Cell Candidate
    o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o) -> Output gate
    c_t = f_t ⊙ c_{t-1} + g_t -> Cell state update
    h_t = o_t ⊙ tanh(c_t) -> hidden state

    Keys:
    ∂c_t/∂c_{t-1} = f_t (controlled by forget gate)
    Constant Error Carousel: gradient maybe make flow unchanged if f_t ≈ 1
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            bias: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size


        self.weight_ih = nn.Parameter(
            torch.Tensor(4 * hidden_size, input_size)
        )
        self.weight_hh = nn.Parameter(
            torch.Tensor(4 * hidden_size, hidden_size)
        )

        if bias:
            self.bias_ih = nn.Parameter(
                torch.Tensor(4 * hidden_size)
            )
            self.bias_hh = nn.Parameter(
                torch.Tensor(4 * hidden_size)
            )
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.reset_parameters()


    
    def reset_parameters(self):
        """
        Xavier initialization with forget gate bias = 1
        Forget gate bias = 1 help model "remember" by default
        """

        stdv = 1.0 / math.sqrt(self.hidden_size)

        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

        # Initialize forget gate bias to 1
        if self.bias_ih is not None:
            # Forget gate is the first chunk
            self.bias_ih.data[0:self.hidden_size].fill_(1.0)
            self.bias_hh.data[0:self.hidden_size].fill_(1.0)


    
    def forward(
            self,
            input: torch.Tensor,
            state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Forward pass for 1 timestamp

        Args:
            input: [batch_size, input_size]
            state: (h,c) tumple
                h: [batch_size, hidden_size]
                c: [batch_size, hidden_size]

        Returns:
            new_h: [batch_size, hidden_size]
            new_c: [batch_size, hidden_size]
        """

        h_prev, c_prev = state

        # Compute all gates at once
        gates = F.linear(
            input,
            self.weight_ih,
            self.bias_ih
        ) + F.linear(
            h_prev,
            self.weight_hh,
            self.bias_hh
        )


        # Split gates: [forget, input, candidate, output]
        f, i, g, o = gates.chunk(4, dim=1)

        # Apply activations
        f = torch.sigmoid(f) # Forget gate: 0 - forget, 1 - remember
        i = torch.sigmoid(i) # Input gate: 0 - ignore, 1 - accept
        g = torch.tanh(g) # Cell candidate: new information
        o = torch.sigmoid(o) # Output gate: control exposure

        # Update cell state (additive update - key for gradient flow)
        c_new = f * c_prev + i * g

        # Update hidden state
        h_new = torch.tanh(c_new) * o

        return h_new, c_new
    



    def compute_gradient_flow(
            self,
            c_prev: torch.Tensor,
            c_new: torch.Tensor,
            forget_gate: torch.Tensor
    ) -> dict:
        """
        Analyze gradient flow through cell state

        ∂L/∂c_{t-1} = ∂L/∂c_t * ∂c_t/∂c_{t-1}
        ∂c_t/∂c_{t-1} = f_t (forget gate)

        Different with RNN:
        - RNN: gradient through Tanh and Weight Matrix
        - LSTM: gradient through Multiplication and Forget Gate (Controlled)
        """


        # torch.no_grad(): just mensure, not backprop

        with torch.no_grad():
            # Gradient flow be controlled by forget gate
            gradient_multiplier = forget_gate.mean().item()

            # Cell state magnitude
            c_prev_norm = c_prev.norm().item()
            c_new_norm = c_new.norm().item()

            # Information retention
            retention = (c_new * c_prev).sum() / (c_prev_norm * c_new_norm + 1e-8)
            retention = retention.item()


        return {
            'gradient_multiplier': gradient_multiplier,  # Average forget gate value
            'cell_norm_before': c_prev_norm,
            'cell_norm_after': c_new_norm,
            'information_retention': retention,  # Cosine similarity
        }