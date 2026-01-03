import torch
import torch.nn as nn
from lstm_cell import LSTMCell
from typing import Optional, Tuple
import torch.nn.functional as F
import numpy as np



class LSTM(nn.Module):

    """
    Multi Layer LSTM

    Advantages over Vanilla RNN
    1. Constant Error Carousel: gradient maybe flow through many timesteps
    2. Gating Mechanism : learn the way when remember when forget
    3. Stable training with long sequences (>100 timesteps)
    4. Maybe capture long-term dependencies

    Tradeoffs:
    1. 4x paramter than RNN (4 gates)
    2. Slower training
    3. More complexity for analyze and interpret
    """

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

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers


        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Stack of LSTM Cells
        self.lstm_cells = nn.ModuleList(
            [
                LSTMCell(
                    embedding_dim if i == 0 else hidden_dim,
                    hidden_dim
                ) for i in range(num_layers)
            ]
        )

        self.dropout = dropout
        
        # Weight tying
        if tie_weights:
            if embedding_dim != hidden_dim:
                print("Warning: Cannot tie weights when embedding_dim != hidden_dim")
            else:
                self.fc.weight = self.embedding.weight


        # Tracking
        self.gate_statistics = []
        self.cell_state_history = []


    
    def forward(
            self,
            x: torch.Tensor, 
            hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass to sequence

        Args:
            x: [batch_size, seq_len]
            hidden: tyuple of (h,c)
                h: [num_layers, batch_size, hidden_dim]
                c: [num_layers, batch_size, hidden_dim]

        Returns:
            output: [batch_size, seq_len, vocab_size]
            hidden: tuple of (h,c)
        """

        batch_size, seq_len = x.shape

        if hidden is None:
            hidden = self.init_hidden(batch_size)

        h,c = hidden

        # Embedding
        embedded = self.embedding(x) # [batch, seq_len, emb_dim]

        outputs = []
        self.gate_statistics = []
        self.cell_state_history = []


        # Process sequence
        for t in range(seq_len):

            input_t = embedded[:, t, :]

            for layer in range(self.num_layers):

                h[layer], c[layer] = self.lstm_cells[layer](input_t, (h[layer], c[layer]))

                # Track cell states
                if layer == self.num_layers - 1:
                    self.cell_state_history.append(c[layer].detach())

                if self.dropout and layer < self.num_layers - 1:
                    input_t = self.dropout(h[layer])
                else:
                    input_t = h[layer]


            # Output
            output_t = self.fc(h[-1])
            outputs.append(output_t)

        
        output = torch.stack(outputs, dim=1)

        return output, (h,c)





    def init_hidden(
        self,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """Initialize hidden and cell states"""
        device = next(self.parameters()).device
        h = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_dim,
            device=device
        )
        c = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_dim,
            device=device
        )

        return (h,c)
    


    def analyze_gates(
            self,
            sequence_input: torch.Tensor
    ) -> dict:
        
        with torch.no_grad():

            batch_size, seq_len = sequence_input.shape
            hidden = self.init_hidden(batch_size)
            h,c = hidden

            forget_gates  = []
            input_gates = []
            output_gates = []

            embedded = self.embedding(sequence_input)

            for t in range(seq_len):

                input_t = embedded[:, t, :]

                for layer in range(self.num_layers):
                    # Compute gates
                    gates = F.linear(
                        input_t,
                        self.lstm_cells[layer].weight_ih
                    ) + F.linear(
                        h[layer],
                        self.lstm_cells[layer].weight_hh
                    )

                    f, i, g, o = gates.chunk(4, dim=1)

                    f = torch.sigmoid(f)
                    i = torch.sigmoid(i)
                    g = torch.tanh(g)
                    o = torch.sigmoid(o)


                    if layer == self.num_layers - 1:
                        forget_gates.append(f.mean().item())
                        input_gates.append(i.mean().item())
                        output_gates.append(o.mean().item())

                    c[layer] = f * c[layer] + i * o
                    h[layer] = o * torch.tanh(c[layer])
                    input_t = h[layer]

        return {
            'mean_forget_gate': float(np.mean(forget_gates)),
            'mean_input_gate': float(np.mean(input_gates)),
            'mean_output_gate': float(np.mean(output_gates)),
            'forget_gate_std': float(np.std(forget_gates)),
            # High forget gate = remembering information
            'remembering_ratio': float(np.mean(np.array(forget_gates) > 0.5)),
        }