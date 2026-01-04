import torch
import torch.nn as nn
from domain.model.base import BaseSequenceModel


class SimpleRNNModel(BaseSequenceModel):

    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            hidden_dim: int,
            num_layers: int = 1,
            **kwargs
    ):
        
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            **kwargs
        )

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )



    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_dim
        )
    

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x = self.dropout(x)


        output, hidden = self.rnn(x, hidden)
        logits = self.output(output)

        return logits, hidden
