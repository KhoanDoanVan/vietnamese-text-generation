import torch
import numpy as np


class GateBehaviorAnalyzer:

    def __init__(self, model, preprocessor, device):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device


    def run(self, sequences):
        if not hasattr(self.model, "analyze_gates"):
            return {
                "error": "Model does not have gates (not LSTM)"
            }
        
        all_stats = []

        for text in sequences:
            ids = self.preprocessor.encode(text, add_special_tokens=False)
            x = torch.tensor([ids], dtype=torch.long).to(self.device)

            gate_stats = self.model.analyze_gates(x)
            all_stats.append(gate_stats)


        aggregated = {}

        for key in all_stats[0].keys():
            values = [
                s[key] for s in all_stats
            ]
            aggregated[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values)
            }

        return aggregated