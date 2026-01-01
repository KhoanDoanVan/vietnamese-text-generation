import torch
import torch.nn as nn
from typing import Optional, Tuple
from vanilla_rnn_cell import VanillaRNNCell


class VanillaRNN(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        tie_weights: bool = True      
    ):
        
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers 
        """
        nn.Embedding:
        Map token id → vector dense
        (batch, time) → (batch, time, embedding_dim)
        """
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        """
        nn.ModuleList:
        - track parameters
        - autograd when throughing
        """
        self.cells = nn.ModuleList(
            [
                VanillaRNNCell(
                    # if the first layer -> input is weight embedding
                    embedding_dim if i == 0 else hidden_dim,
                    hidden_dim
                ) for i in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.output = nn.Linear(hidden_dim, vocab_size)
        # Wheather layers are using the same weight or not
        if tie_weights and embedding_dim == hidden_dim:
            self.output.weight = self.embedding.weight



    def init_hidden(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
    

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        b: batch_size
        t: timesteps (sequence length)
        """
        b, t = x.shape
        device = x.device

        if hidden is None:
            hidden = self.init_hidden(b, device)

        emb = self.embedding(x)
        outputs = []

        """
        - Don't parallel timestamp
        - Timestep-by-Timestep
        """
        for step in range(t):
            inp = emb[:, step]
            # Loop layers RNN
            """
            time step t:
            input → RNN layer 0 → RNN layer 1 → ... → RNN layer N
            """
            for layer, cell in enumerate(self.cells):
                hidden[layer] = cell(inp, hidden[layer])
                inp = (
                    # Dropout each hidden layer Except the last layer
                    self.dropout(hidden[layer])
                    if self.dropout and layer < self.num_layers - 1 else hidden[layer]
                )
            # Get hidden of the last layer
            outputs.append(self.output(hidden[-1]))

        """
        Each loop time:
        - outputs = [(b, vocab),(b, vocab),...t times] -> list Python - not a Tensor

        ===> Stack: t x (b, vocab) -> (b, t, vocab) (dim=0 -> batch, dim=1 -> time, dim2 -> vocab/feature)        
        """
        return torch.stack(outputs, dim=1), hidden
    