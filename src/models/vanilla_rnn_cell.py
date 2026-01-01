import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VanillaRNNCell(nn.Module):

    """
    h_t = Tanh( W_ih x x_t + W_hh x h_{t-1} + b )
    """


    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            bias: bool = True
    ):
        
        super().__init__()
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(
            torch.empty(hidden_size, input_size)
        )
        self.weight_hh = nn.Parameter(
            torch.empty(hidden_size, input_size)
        )
        self.bias = nn.Parameter(
            torch.empty(hidden_size)
        ) if bias else None

        self.reset_parameters()

    
    def reset_parameters(self):
        """
        Initialize W with random range(-bound, bound)
        """
        bound = 1.0 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.weight_ih, -bound, bound)
        nn.init.uniform_(self.weight_hh, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(
            F.linear(x, self.weight_ih, self.bias) + F.linear(h, self.weight_hh)
        )
