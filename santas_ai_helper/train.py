import argparse

import lightning.pytorch as pl
import torch

from saih.data import DataModule
from saih.model import Model


def main():
    args = _get_args()

    datamodule = DataModule(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )

    model = Model(
        learning_rate=args.learning_rate,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        max_epochs=args.epochs,
    )

    trainer.fit(model, datamodule)


def _get_args():
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the directory containing the data.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="The batch size to use for training, validation, and testing.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        required=True,
        help="The learning rate to use for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="The number of epochs to use for training.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
