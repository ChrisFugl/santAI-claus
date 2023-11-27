import argparse
from random import random

import lightning.pytorch as pl
import torch

from saih.data import DataModule
from saih.model import Model


def main():
    args = _get_args()

    datamodule = DataModule(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        filter=_create_filter(args.min_age, args.max_age),
    )

    model = Model(
        learning_rate=args.learning_rate,
    )

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="val/f1",
            mode="max",
            save_top_k=1,
            save_last=True,
            filename="best",
        ),
    ]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        max_epochs=args.epochs,
        log_every_n_steps=5,
        precision=16,
        callbacks=callbacks,
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
    parser.add_argument(
        "--min-age",
        type=int,
        required=False,
        default=0,
        help="The minimum age of people included in the dataset.",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        required=False,
        default=18,
        help="The maximum age of people included in the dataset.",
    )
    parser.add_argument(
        "--drop-person-prop",
        type=float,
        required=False,
        default=0.0,
    )

    args = parser.parse_args()

    return args


def _create_filter(min_age: int, max_age: int, drop_person_prop: float):
    def _filter(person: dict):
        return random() < drop_person_prop or (min_age <= person["age"] and person["age"] <= max_age)

    return _filter


if __name__ == "__main__":
    main()
