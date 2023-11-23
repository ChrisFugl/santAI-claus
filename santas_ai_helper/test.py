import argparse

import lightning.pytorch as pl
import torch

from saih.data import DataModule
from saih.ensemble_model import EnsembleModel


def main():
    args = _get_args()

    datamodule = DataModule(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )

    model = EnsembleModel(args.model)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        precision=16,
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    if len(args.model) == 0:
        raise ValueError("At least one model must be specified.")

    return args


if __name__ == "__main__":
    main()
