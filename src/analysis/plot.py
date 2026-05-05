import os
import numpy as np
import matplotlib.pyplot as plt


def save_error_plots(report: dict, output_dir: str) -> None:
    """Save temporal RMSE curves for u, v, p to a single PNG."""
    os.makedirs(output_dir, exist_ok=True)
    fields = ["u", "v", "p"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, field in zip(axes, fields):
        temporal = report.get(f"temporal_rmse_{field}")
        rmse = report.get(f"rmse_{field}", float("nan"))
        rel = report.get(f"rel_err_{field}", float("nan"))
        if temporal is not None:
            ax.plot(temporal, linewidth=1.2)
        ax.set_title(f"{field}   RMSE={rmse:.4f}  RelErr={rel:.4f}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "temporal_rmse.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def save_summary_table(report: dict, output_dir: str) -> None:
    """Print and save a plain-text metrics summary table."""
    os.makedirs(output_dir, exist_ok=True)
    header = f"{'Field':<8}{'RMSE':>12}{'Rel. Error':>14}"
    sep = "-" * len(header)
    rows = [header, sep]
    for field in ["u", "v", "p"]:
        rmse = report.get(f"rmse_{field}", float("nan"))
        rel = report.get(f"rel_err_{field}", float("nan"))
        rows.append(f"{field:<8}{rmse:>12.6f}{rel:>14.6f}")
    table = "\n".join(rows)
    print(table)

    path = os.path.join(output_dir, "metrics_summary.txt")
    with open(path, "w") as f:
        f.write(table + "\n")
    print(f"Saved: {path}")
