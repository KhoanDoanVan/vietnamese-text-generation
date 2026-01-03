from dataclasses import dataclass


@dataclass
class EpochMetrics:
    loss: float
    perplexity: float
    grad_mean: float
    grad_std: float
    grad_max: float
    epoch_time: float