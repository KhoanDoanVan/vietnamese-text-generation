import torch
import numpy as np
from utils.torch_utils import init_hidden_if_needed



class GradientFlowAnalyzer:

    def __init__(self, model, preprocessor, device):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device


    def run(self, sequence_lengths):

        results = {}

        for seq_len in sequence_lengths:
            dummy = torch.randint(
                0, self.preprocessor.vocab_size,
                (1, seq_len)
            ).to(self.device)

            self.model.zero_grad()

            hidden = init_hidden_if_needed(
                self.model,
                1,
                self.device
            )

            outputs,_ = (
                self.model(dummy, hidden)
                if hidden else self.model(dummy)
            )

            loss = outputs[:, -1].sum()
            loss.backward()

            grads = [
                p.grad.norm().item()
                for n,p in self.model.named_parameters()
                if p.grad is not None and "weight" in n
            ]

            results[seq_len] = {
                "mean_grad_norm": np.mean(grads),
                "max_grad_norm": np.max(grads),
                "min_grad_norm": np.min(grads),
                "grad_norms": grads
            }

            if hasattr(self.model, "analyze_gradient_flow"):
                results[seq_len].update(
                    self.model.analyze_gradient_flow(seq_len)
                )

        return results