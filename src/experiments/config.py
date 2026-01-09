import torch
import yaml

class ExperimentConfig:

    def __init__(self, cfg: dict):

        # Data settings
        self.data = cfg["data"]

        # Model settings
        self.model = cfg["model"]

        # Training settings
        self.training = cfg["training"]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Evaluation settings
        self.evaluations = cfg["evaluation"]

        # Paths
        self.paths = cfg["paths"]

    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        return cls(cfg)