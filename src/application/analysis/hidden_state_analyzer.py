import numpy as np
import torch
from typing import Dict, List



class HiddenStateAnalyzer:

    def analyze(
            self,
            hidden_states: List[torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        
        norms, means, stds = [], [], []


        for h in hidden_states:
            h = h.detach().cpu().numpy()
            norms.append(np.linalg.norm(h, axis=1).mean())
            means.append(h.mean())
            stds.append(h.std())



        return {
            "norms": np.array(norms),
            "means": np.array(means),
            "stds": np.array(stds)
        }