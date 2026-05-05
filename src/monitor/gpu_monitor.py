import csv
import os
import threading
from datetime import datetime
from typing import Callable, List, Optional


class GPUMonitor:
    """
    Samples GPU metrics in a background thread and writes to a CSV file.

    Tries pynvml first; falls back to nvidia-smi subprocess if pynvml is unavailable.
    Fails silently if no GPU is present — the CSV will just contain the header row.
    """

    def __init__(self, log_dir: str, interval_sec: float = 5.0) -> None:
        self.log_dir = log_dir
        self.interval_sec = interval_sec
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.csv_path: Optional[str] = None

    def start(self, run_name: Optional[str] = None) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gpu_{run_name}_{ts}.csv" if run_name else f"gpu_{ts}.csv"
        self.csv_path = os.path.join(self.log_dir, filename)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=15)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        reader = self._make_reader()
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp", "gpu_util_pct", "mem_used_mb", "mem_total_mb", "temp_c", "power_w"]
            )
            while not self._stop_event.is_set():
                row = reader()
                if row is not None:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    writer.writerow([ts] + row)
                    f.flush()
                self._stop_event.wait(self.interval_sec)

    def _make_reader(self) -> Callable[[], Optional[List]]:
        """Returns a callable that reads one sample of GPU stats."""
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            def read_pynvml() -> Optional[List]:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    try:
                        power = round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, 1)
                    except pynvml.NVMLError:
                        power = -1
                    return [
                        util.gpu,
                        mem.used // (1024 * 1024),
                        mem.total // (1024 * 1024),
                        temp,
                        power,
                    ]
                except Exception:
                    return None

            return read_pynvml
        except Exception:
            pass

        import subprocess

        def read_smi() -> Optional[List]:
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return [v.strip() for v in result.stdout.strip().split(",")]
            except Exception:
                return None

        return read_smi
