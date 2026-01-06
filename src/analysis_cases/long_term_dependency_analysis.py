import numpy as np
import torch
from utils.torch_utils import init_hidden_if_needed


class LongTermDependencyAnalyzer:

    def __init__(self, model, preprocessor, device):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device

    
    def run(self, sequences, distances):
        results = {d: [] for d in distances}

        for text in sequences:
            ids = self.preprocessor.encode(text, add_special_tokens=False)
            if len(ids) <= max(distances):
                continue


            x = torch.tensor([ids]).to(self.device)
            hidden = init_hidden_if_needed(self.model, 1, self.device)


            with torch.no_grad():
                self.model(x, hidden)

            if not hasattr(self.model, "hidden_states_history"):
                continue

            hs = self.model.history_states_history

            for d in distances:
                for t in range(d, len(hs)):
                    h1 = hs[t].cpu().numpy().ravel()
                    h2 = hs[t - d].cpu().numpy().ravel()
                    sim = np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-8)
                    results[d].append(sim)


        return {
            d: {
                "mean": np.mean(v) if v else 0,
                "std": np.std(v) if v else 0,
                "median": np.median(v) if v else 0
            } for d, v in results.items()
        }