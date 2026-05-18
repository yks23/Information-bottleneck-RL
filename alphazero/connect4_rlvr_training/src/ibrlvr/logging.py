from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class ExperimentLogger:
    def __init__(self, run_dir: Path, config: dict[str, Any]) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._fh = self.metrics_path.open("a", encoding="utf-8")
        self.wandb_run = None

        wandb_cfg = config.get("wandb", {})
        mode = os.environ.get("WANDB_MODE", wandb_cfg.get("mode", "offline"))
        if mode != "disabled":
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=wandb_cfg.get("project", "information-bottleneck-alphazero"),
                    name=config.get("run_name"),
                    config=config,
                    dir=str(run_dir),
                    mode=mode,
                )
            except Exception as exc:  # pragma: no cover - depends on external service.
                print(f"wandb disabled after init failure: {exc}")
                self.wandb_run = None

    def log(self, step: int, metrics: dict[str, float]) -> None:
        payload = {"step": step, **metrics}
        self._fh.write(json.dumps(payload, sort_keys=True) + "\n")
        self._fh.flush()
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def close(self) -> None:
        self._fh.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()
