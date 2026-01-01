from typing import Dict, List
import torch
import numpy as np


class GradientMonitor:

    def __init__(self):
        self.norms: List[float] = []



    def attach(self, model: torch.nn.Module) -> None:
        def hook(_, __, grad_output):
            if grad_output and grad_output[0] is not None:
                self.norms.append(grad_output[0].norm().item())


        for m in model.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
                m.register_full_backward_hook(hook)


    def stats(self) -> Dict[str, float]:
        if not self.norms:
            return {}
        

        g = np.array(self.norms)

        return {
            "mean": float(g.mean()),
            "std": float(g.std()),
            "min": float(g.min()),
            "max": float(g.max()),
            "vanishing_ratio": float((g < 1e-7).mean()),
            "exploding_ratio": float((g > 10).mean())
        }
    


    def reset(self):
        self.norms.clear()
