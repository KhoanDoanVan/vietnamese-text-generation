import torch
import torch.nn as nn


class Perplexity:

    def __init__(self, ignore_index: int = -100):
        self.criterion = nn.CrossEntropyLoss(
            reduction="none",
            ignore_index=ignore_index
        )



    def compute(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            mask: torch.Tensor | None = None
    ) -> float:
        
        # b: batch_size
        # t: sequence_length
        # v: vocab size
        b, t, v = logits.shape

        # Calculate loss for each token
        loss = self.criterion(
            logits.view(-1, v),
            targets.view(-1)
        ).view(b, t)

        if mask is not None:

            # Normal mask almost is: 1 for valid, 0 for padding
            loss = (loss * mask).sum() / mask.sum()

        else:

            # calculate for entire token
            loss = loss.mean()

        # Transform loss -> perplexity

        """
        Example for Perplexity:
        - vocab: 1000 words
        - model random -> probability is 1/1000
        -> loss = -log(1/1000) = 6.9
        -> perplexity = e^loss = 1000 
        ===> Perplexity = how many words which model confused?
        """
        return torch.exp(loss).item()

