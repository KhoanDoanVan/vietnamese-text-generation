from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingHistory:
    train_loss: List[float] = field(default_factory=list)
    train_ppl: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_ppl: List[float] = field(default_factory=list)
    grad_norm: List[float] = field(default_factory=list)
    epoch_time: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)


    def add_epoch(
        self,
        train_loss: float,
        train_ppl: float,
        val_loss: float,
        val_ppl: float,
        grad_norm: float,
        epoch_time: float,
        lr: float,
    ):
        self.train_loss.append(train_loss)
        self.train_ppl.append(train_ppl)
        self.val_loss.append(val_loss)
        self.val_ppl.append(val_ppl)
        self.grad_norm.append(grad_norm)
        self.epoch_time.append(epoch_time)
        self.lr.append(lr)