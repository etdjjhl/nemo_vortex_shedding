# Quantitative analysis of inference results.
# Loads the trained model, runs rollout on the test set, computes error metrics,
# and saves plots to ./analysis/

import hydra
from omegaconf import DictConfig

from physicsnemo.utils.logging import PythonLogger

from inference import MGNRollout
from src.analysis.metrics import compute_metrics_report
from src.analysis.plot import save_error_plots, save_summary_table


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = PythonLogger("analyze")
    logger.file_logging()
    logger.info("Analysis started...")

    rollout = MGNRollout(cfg, logger)
    rollout.predict()

    report = compute_metrics_report(rollout.pred, rollout.exact)
    save_error_plots(report, output_dir="./analysis")
    save_summary_table(report, output_dir="./analysis")

    logger.info("Analysis complete. Results saved to ./analysis/")


if __name__ == "__main__":
    main()
