from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class GradientFlowResult:
    mean_grad_norm: float
    max_grad_norm: float
    min_grad_norm: float
    grad_norms: float