import json
import re
from pathlib import Path
from shutil import rmtree
from tempfile import gettempdir
from typing import Union

import lightning.pytorch as pl
import torch
from transformers import BertTokenizer

from saih.constants import MODEL_NAME


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_dir: Union[str, Path],
        train_val_test_split: tuple[float, float, float] = (0.8, 0.1, 0.1),
        niceness_threshold: Union[int, float] = 0,
    ):
        """
        Args:
            batch_size: The batch size to use for training, validation, and testing.
            data_dir: The directory containing the data.
            train_val_test_split: The split to use for training, validation, and testing. Must sum to 1.
            niceness_threshold: The threshold to use to determine whether a person is naughty or nice
                A person is classified as naughty if their niceness score is below the threshold.
        """
        super().__init__()

        if abs(sum(train_val_test_split) - 1) > 1e-10:
            raise ValueError(
                "The train_val_test_split must sum to 1. "
                + f"Got {train_val_test_split}."
            )

        self._batch_size = batch_size
        self._data_dir = Path(data_dir)
        self._niceness_threshold = niceness_threshold

        self._prepared_data_dir = Path(gettempdir()) / "santas_ai_helper"

        self.prepare_data_per_node = True

    def prepare_data(self):
        self._prepare_split("training")
        self._prepare_split("validation")
        self._prepare_split("testing")

    def setup(self, stage: str):
        if stage == "fit":
            self._train_dataset = Dataset(self._prepared_data_dir / "training")
            self._val_dataset = Dataset(self._prepared_data_dir / "validation")
        elif stage == "test":
            self._test_dataset = Dataset(self._prepared_data_dir / "testing")
        else:
            raise ValueError(f"Unsupported stage: {stage}")

    def teardown(self, stage: str):
        if stage == "fit":
            del self._train_dataset
            del self._val_dataset
            rmtree(self._prepared_data_dir / "training")
            rmtree(self._prepared_data_dir / "validation")
        elif stage == "test":
            del self._test_dataset
            rmtree(self._prepared_data_dir / "testing")
        else:
            raise ValueError(f"Unsupported stage: {stage}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            collate_fn=Collator(),
            num_workers=0,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            collate_fn=Collator(),
            num_workers=0,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            collate_fn=Collator(),
            num_workers=0,
            shuffle=False,
            pin_memory=True,
        )

    def _prepare_split(self, split: str):
        data: list[tuple[str, int]] = []

        # Filter data
        split_dir = self._data_dir / split
        paths = split_dir.glob("*.json")
        for path in paths:
            try:
                person = json.loads(path.read_text())
            except json.JSONDecodeError:
                # Some of the files were corrupted by the generator (it generated invalid JSON).
                # We could fix them, but it is easier to generate more data and ignore the corrupted files.
                continue

            if "description" not in person:
                # We may encounter people in the directory for whom we haven't yet generated descriptions.
                # We will just ignore these people for now.
                continue

            description = person["description"]
            label = int(person["score"] >= self._niceness_threshold)

            try:
                description = self._extract_enumerated_lines(description)
            except ValueError:
                # We only want to keep people who follow the "1, 2, ..." format.
                continue

            data.append((description, label))

        print(f"{split} size: {len(data):,}")

        save_dir = self._prepared_data_dir / split
        save_dir.mkdir(parents=True, exist_ok=True)
        for index, (description, label) in enumerate(data):
            output = {"description": description, "label": label}
            path = save_dir / f"{index}.json"
            path.write_text(json.dumps(output))

    def _extract_enumerated_lines(self, description: str) -> bool:
        """
        Llama was asked to generate descriptions according to the following format:

        -------------
        [description]

        1. [month]: [event]
        2. [month]: [event]
        ...
        n. [month]: [event]
        -------------

        We will only be providing the enumerated lines (1, 2, ..., n) to the model.
        """
        lines = description.split("\n")
        enumerated_lines = []
        number_pattern = re.compile(r"^\d+\.")
        for line in lines:
            if number_pattern.match(line):
                enumerated_lines.append(line)

        if len(enumerated_lines) == 0:
            raise ValueError("No enumerated lines found.")

        return "\n".join(enumerated_lines)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: Union[str, Path]):
        super().__init__()

        self._data_dir = Path(data_dir)

        self._paths = list(self._data_dir.glob("*.json"))

    def __getitem__(self, index: int):
        path = self._paths[index]

        person = json.loads(path.read_text())

        description = person["description"]
        label = person["label"]

        return description, label

    def __len__(self) -> int:
        return len(self._paths)


class Collator:
    def __init__(self):
        self._tokenizer = None

    def __call__(self, batch: list[tuple[str, int]]):
        if self._tokenizer is None:
            self._tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

        descriptions, labels = zip(*batch)

        tokenized_descriptions = self._tokenizer(
            descriptions,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return tokenized_descriptions, torch.tensor(labels)
