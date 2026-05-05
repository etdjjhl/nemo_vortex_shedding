import numpy as np
import torch
from typing import Dict, List


FIELD_NAMES: Dict[int, str] = {0: "u", 1: "v", 2: "p"}


def compute_rmse(
    pred: List[torch.Tensor], exact: List[torch.Tensor], field_idx: int
) -> float:
    """Overall RMSE for one field across all time steps."""
    mse_per_step = [
        (p[:, field_idx] - e[:, field_idx]).pow(2).mean().item()
        for p, e in zip(pred, exact)
    ]
    return float(np.sqrt(np.mean(mse_per_step)))


def compute_temporal_rmse(
    pred: List[torch.Tensor], exact: List[torch.Tensor], field_idx: int
) -> np.ndarray:
    """RMSE at each time step for one field. Returns array of shape (T,)."""
    return np.array(
        [
            float(np.sqrt((p[:, field_idx] - e[:, field_idx]).pow(2).mean().item()))
            for p, e in zip(pred, exact)
        ]
    )


def compute_relative_error(
    pred: List[torch.Tensor], exact: List[torch.Tensor], field_idx: int
) -> float:
    """Mean relative L2 error across all time steps."""
    errors = []
    for p, e in zip(pred, exact):
        num = (p[:, field_idx] - e[:, field_idx]).pow(2).sum().item()
        den = e[:, field_idx].pow(2).sum().item()
        if den > 1e-10:
            errors.append(num / den)
    return float(np.sqrt(np.mean(errors))) if errors else float("nan")


def compute_metrics_report(
    pred: List[torch.Tensor], exact: List[torch.Tensor]
) -> dict:
    """Compute RMSE, relative error, and temporal RMSE for all fields (u, v, p)."""
    report = {}
    for idx, name in FIELD_NAMES.items():
        report[f"rmse_{name}"] = compute_rmse(pred, exact, idx)
        report[f"rel_err_{name}"] = compute_relative_error(pred, exact, idx)
        report[f"temporal_rmse_{name}"] = compute_temporal_rmse(pred, exact, idx)
    return report
