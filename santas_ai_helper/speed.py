import argparse

import lightning.pytorch as pl
import pandas as pd
import torch

from saih.data import DataModule
from saih.ensemble_model import EnsembleModel


class TimingCallback(pl.Callback):
    def on_test_start(self, trainer, pl_module):
        self.times = []

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.end.record()
        torch.cuda.synchronize()
        self.times.append(self.start.elapsed_time(self.end))

    def on_test_end(self, trainer, pl_module):
        timings = pd.DataFrame({"time_ms": self.times})
        timings.to_csv("timings.csv", index=False)


def main():
    args = _get_args()

    datamodule = DataModule(batch_size=1, data_dir=args.data_dir)

    model = EnsembleModel(args.model)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        precision=16,
        callbacks=[TimingCallback()],
    )

    trainer.test(model, datamodule)


def _get_args():
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the directory containing the data.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        nargs="+",
    )

    args = parser.parse_args()

    if len(args.model) == 0:
        raise ValueError("At least one model must be specified.")

    return args


if __name__ == "__main__":
    main()
