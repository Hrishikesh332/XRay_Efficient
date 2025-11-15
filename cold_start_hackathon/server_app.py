from logging import INFO
import os

import torch
import wandb
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.common import log
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from cold_start_hackathon.task import Net
from cold_start_hackathon.util import (
    compute_aggregated_metrics,
    log_training_metrics,
    log_eval_metrics,
    save_best_model,
)

# ============================================================================
# W&B Configuration - Fill in your credentials or set via environment variables
# ============================================================================
# Option 1: Set these constants directly (e.g., WANDB_API_KEY = "your_api_key_here")
# Option 2: Leave as None and set environment variables (WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT)
# If all W&B config is None/unset, W&B logging will be disabled
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", None)  # Your W&B API key
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", None)  # Your W&B project name
# ============================================================================

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    # Log GPU device
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    log(INFO, f"Device: {device}")

    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    local_epochs: int = context.run_config["local-epochs"]

    # Get run name from environment variable (set by submit-job.sh). Feel free to change this.
    run_name = os.environ.get("JOB_NAME", "your_custom_run_name")

    # Initialize W&B if credentials are provided
    use_wandb = WANDB_API_KEY and WANDB_PROJECT
    if use_wandb:
        wandb.login(key=WANDB_API_KEY)
        log(INFO, "Wandb login successful")
        wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            config={
                "num_rounds": num_rounds,
                "learning_rate": lr,
                "local_epochs": local_epochs,
            }
        )
        log(INFO, "Wandb initialized with run_id: %s", wandb.run.id)
    else:
        log(INFO, "W&B disabled (credentials not provided). Set WANDB_API_KEY, WANDB_ENTITY, and WANDB_PROJECT to enable.")

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    strategy = HackathonFedAvg(fraction_train=1, run_name=run_name)
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    log(INFO, "Training complete")
    if use_wandb:
        wandb.finish()
        log(INFO, "Wandb run finished")


class HackathonFedAvg(FedAvg):
    """FedAvg strategy that logs metrics and saves best model to W&B."""

    def __init__(self, *args, run_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._best_auroc = None
        self._run_name = run_name or "your_run"

    def aggregate_train(self, server_round, replies):
        arrays, metrics = super().aggregate_train(server_round, replies)
        self._arrays = arrays
        log_training_metrics(replies, server_round)
        return arrays, metrics

    def aggregate_evaluate(self, server_round, replies):
        agg_metrics = compute_aggregated_metrics(replies)
        log_eval_metrics(replies, agg_metrics, server_round, self.weighted_by_key, lambda msg: log(INFO, msg))
        self._best_auroc = save_best_model(self._arrays, agg_metrics, server_round, self._run_name, self._best_auroc)
        return agg_metrics
