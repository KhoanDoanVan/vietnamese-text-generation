from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class BenchmarkResult:
    parameters: Dict[str, Any]
    speed: Dict[int, Dict[str, float]]