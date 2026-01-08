import time
import torch
from typing import Dict, List



class BenchmarkAnalyzer:

    def __init__(self, model, preprocessor, device):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device


    def run(
            self,
            sequence_lengths: List[int] = [10, 50, 100, 200]
    ) -> Dict:
        
        results = {}

        total_params = sum(p.numel() for p in self.model.parameters())

        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        results["parameters"] = {
            "total": total_params,
            "trainable": trainable_params,
            "size_mb": total_params * 4 / (1024 ** 2) # float32
        }

        # Speed benchmark
        speed_results = {}

        for seq_len in sequence_lengths:

            dummy_input = torch.randint(
                0,
                self.preprocessor.vocab_size,
                (1, seq_len)
            ).to(self.device)

            # -------- Warm up --------
            for _ in range(10):
                with torch.no_grad():
                    if hasattr(self.model, "init_hidden"):
                        hidden = self.model.init_hidden(1)
                        if isinstance(hidden, tuple):
                            hidden = (
                                hidden[0].to(self.device),
                                hidden[1].to(self.device),
                            )
                        else:
                            hidden = hidden.to(self.device)

                        self.model(dummy_input, hidden)
                    else:
                        self.model(dummy_input)

            # -------- Measure forward pass --------
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.time()

            for _ in range(100):
                with torch.no_grad():
                    if hasattr(self.model, "init_hidden"):
                        hidden = self.model.init_hidden(1)
                        if isinstance(hidden, tuple):
                            hidden = (
                                hidden[0].to(self.device),
                                hidden[1].to(self.device),
                            )
                        else:
                            hidden = hidden.to(self.device)

                        self.model(dummy_input, hidden)
                    else:
                        self.model(dummy_input)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.time()

            avg_time = (end_time - start_time) / 100

            speed_results[seq_len] = {
                "forward_time_ms": avg_time * 1000,
                "tokens_per_second": seq_len / avg_time
            }

        results["speed"] = speed_results
        return results