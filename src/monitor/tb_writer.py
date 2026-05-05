import os
from torch.utils.tensorboard import SummaryWriter


class TBWriter:
    """Thin wrapper around SummaryWriter for per-epoch training metrics."""

    def __init__(self, log_dir: str) -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_epoch(self, epoch: int, loss: float, lr: float = None) -> None:
        self.writer.add_scalar("train/loss", loss, epoch)
        if lr is not None:
            self.writer.add_scalar("train/lr", lr, epoch)

    def close(self) -> None:
        self.writer.close()
