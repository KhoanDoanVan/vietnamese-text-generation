import torch
import numpy as np
from utils.torch_utils import init_hidden_if_needed


class PerplexityAnalyzer:

    def __init__(self, model, preprocessor, device, window_size=50):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.window_size = window_size
        self.criterion = torch.nn.CrossEntropyLoss()


    def run(self, sequences):

        records = []

        for text in sequences:

            ids = self.preprocessor.encode(text, add_special_tokens=False)
            if len(ids) < self.window_size + 1:
                continue

            for start in range(0, len(ids) - self.window_size, self.window_size // 2):

                x = ids[start: start + self.window_size]
                y = ids[start + 1: start + self.window_size + 1]

                x_t = torch.tensor([x]).to(self.device)
                y_t = torch.tensor([y]).to(self.device)

                with torch.no_grad():
                    hidden = init_hidden_if_needed(self.model, 1, self.device)
                    outputs, _ = (
                        self.model(x_t, hidden)
                        if hidden else self.model(x_t)
                    )

                    logits = outputs.view(-1, outputs.size(-1))
                    targets = y_t.view(-1)

                    loss = self.criterion(logits, targets)
                    ppl = torch.exp(loss).item()


                records.append({
                    "position": start,
                    "perplexity": ppl
                })

        self._aggregate(records)


    def _aggregate(self, records):
        ranges = [0, 50, 100, 200, 500]
        stats = {}

        for i in range(len(range) - 1):
            s, e = ranges[i], ranges[i + 1]
            values = [
                r["perplexity"] for r in records if s <= r["positon"] < e
            ]
            if values:
                stats[f"{s}-{e}"] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "count": len(values)
                }

        
        return stats