import argparse
from random import random

import lightning.pytorch as pl
import torch

from saih.constants import MODELS_DIR
from saih.data import DataModule
from saih.model import Model


def main():
    args = _get_args()

    datamodule = DataModule(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        filter=_create_filter(args.min_age, args.max_age, args.drop_person_prop),
    )

    model = Model(
        learning_rate=args.learning_rate,
    )

    output_dir = MODELS_DIR / args.name

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="val/f1",
            mode="max",
            save_top_k=1,
            save_last=True,
            filename="best",
            dirpath=output_dir,
        ),
    ]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        default_root_dir=output_dir,
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
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="The name of the model.",
    )

    args = parser.parse_args()

    return args


def _create_filter(min_age: int, max_age: int, drop_person_prop: float):
    def _filter(person: dict):
        age = person["age"]
        kept_at_random = random() > drop_person_prop
        return kept_at_random and min_age <= age and age <= max_age

    return _filter


if __name__ == "__main__":
    main()
