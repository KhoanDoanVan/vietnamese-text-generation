
import torch
import torch.nn as nn
from torch.nn.utils import clip_grads_with_norm_
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from utils.to_device import _to_device
from domain.epoch_metrics import EpochMetrics
from domain.training_history import TrainingHistory
import time
from tqdm import tqdm
import numpy as np 
from domain.metrics.perplexity import Perplexity


class LanguageModelTrainer:

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            optimizer: Optimizer,
            device: str = "cpu",
            grad_clip: float = 5.0,
            log_interval: int = 100
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip
        self.log_interval = log_interval

        self.ppl_metric = Perplexity(ignore_index=100)
        self.criterion = nn.CrossEntropyLoss()
        self.history = TrainingHistory()

        self.best_val_loss = float("inf")
        self.best_model_state = None


    def _forward(
            self,
            inputs: torch.Tensor
    ) -> torch.Tensor:
        
        if hasattr(self.model, "init_hidden"):
            hidden = self.model.init_hidden(input.size(0))
            hidden = _to_device(hidden, self.device)
            outputs, _ = self.model(inputs, hidden)
        else:
            outputs, _ = self.model(inputs)

        return outputs
    


    def _backward(
            self,
            loss: torch.Tensor
    ) -> float:
        
        loss.backward()
        grad_norm = clip_grads_with_norm_(
            self.model.parameters(),
            self.grad_clip
        )
        self.optimizer.step()

        return grad_norm.item()


    def _compute_loss(
            self,
            outputs: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        logits = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        return self.criterion(logits, targets)
    


    def train_epoch(self, epoch: int) -> EpochMetrics:

        self.model.train()

        total_loss = 0.0
        total_tokens = 0
        total_ppl = 0.0
        grad_norms = []

        start_time = time.time()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for step, (inputs, targets, mask) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            mask = mask.to(self.device) if mask is not None else None

            self.optimizer.zero_grad()

            outputs = self._forward(inputs)
            loss = self._compute_loss(outputs, targets)
            grad_norm = self._backward(loss)

            # --- METRIC (No Track Gradient - Backward) --- 
            with torch.no_grad():
                ppl = self.ppl_metric.compute(
                    logits=outputs,
                    targets=targets,
                    mask=mask
                )
            

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_ppl += ppl * batch_size
            total_tokens += batch_size
            grad_norms.append(grad_norm)

            if step % self.log_interval == 0:
                avg_loss = total_loss / total_tokens
                avg_ppl = total_ppl / total_tokens
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    ppl=f"{avg_ppl:.2f}",
                    grad=f"{grad_norm:.3f}"
                )

        return EpochMetrics(
            loss=(total_loss / total_tokens),
            perplexity=(total_ppl / total_tokens),
            grad_mean=np.mean(grad_norms),
            grad_std=np.std(grad_norms),
            grad_max=np.max(grad_norms),
            epoch_time=(time.time() - start_time)
        )
    


    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:

        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self._forward(inputs)
            loss = self._compute_loss(outputs, targets)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_tokens += batch_size


        avg_loss = total_loss / total_tokens

        return avg_loss, np.exp(avg_loss)
    


    def train(self, num_epochs: int, scheduler=None):
        print(f"Training {num_epochs} epochs")
        print(f"Params: {sum(p.numel() for p in self.model.parameters()):,}")


        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_loss, val_ppl = self.evaluate()


            if scheduler:
                scheduler.step(val_loss)

            self._log_epoch(train_metrics, val_loss, val_ppl, epoch, num_epochs)
            self._save_best_model(epoch, val_loss, val_ppl)

        print("Training completed !!!")



    def _log_epoch(
            self,
            train: EpochMetrics,
            val_loss: float,
            val_ppl: float,
            epoch: int,
            total_epochs: int,
    ):
        self.history.add_epoch(
            train_loss=train.loss,
            train_ppl=train.perplexity,
            val_loss=val_loss,
            val_ppl=val_ppl,
            grad_norm=train.grad_mean,
            epoch_time=train.epoch_time,
            lr=self.optimizer.param_groups[0]["lr"]
        )

        print(f"\nEpoch {epoch}/{total_epochs}")
        print(f" Train: loss={train.loss:.4f}, ppl={train.perplexity:.2f}")
        print(f" Val  : loss={val_loss:.4f}, ppl={val_ppl:.2f}")
        print(f" Grad : {train.grad_mean:.3f} ± {train.grad_std:.3f}")
        print(f" Time : {train.epoch_time:.2f}s")



    def _save_best_model(
            self,
            epoch: int,
            val_loss: int,
            val_ppl: int
    ):
        if val_loss < self.best_model_state:
            self.best_val_loss = val_loss
            self.best_model_state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "val_ppl": val_ppl
            }

            print("====>> New Best Model <<====")