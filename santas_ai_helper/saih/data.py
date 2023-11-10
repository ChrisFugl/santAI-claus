import json
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
        num_workers: int = 3,
    ):
        """
        Args:
            batch_size: The batch size to use for training, validation, and testing.
            data_dir: The directory containing the data.
            train_val_test_split: The split to use for training, validation, and testing. Must sum to 1.
            niceness_threshold: The threshold to use to determine whether a person is naughty or nice
                A person is classified as naughty if their niceness score is below the threshold.
            num_workers: The number of workers to use for loading the data.
        """
        super().__init__()

        if abs(sum(train_val_test_split) - 1) > 1e-10:
            raise ValueError(
                "The train_val_test_split must sum to 1. "
                + f"Got {train_val_test_split}."
            )

        self._batch_size = batch_size
        self._data_dir = Path(data_dir)
        self._train_val_test_split = train_val_test_split
        self._niceness_threshold = niceness_threshold
        self._num_workers = num_workers

        self._prepared_data_dir = Path(gettempdir()) / "santas_ai_helper"

        self.prepare_data_per_node = True

    def prepare_data(self):
        data: list[tuple[str, int]] = []

        # Filter data
        paths = self._data_dir.glob("*.json")
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
            data.append((description, label))

        print(f"Dataset size: {len(data):,}")

        # Split into train, val, and test
        train_ratio, val_ratio, _ = self._train_val_test_split

        train_end_index = int(train_ratio * len(data))
        train_data = data[:train_end_index]
        print(f"Train size: {len(train_data):,}")

        val_end_index = train_end_index + int(val_ratio * len(data))
        val_data = data[train_end_index:val_end_index]
        print(f"Validation size: {len(val_data):,}")

        test_data = data[val_end_index:]
        print(f"Test size: {len(test_data):,}")

        self._save_data(train_data, "train")
        self._save_data(val_data, "val")
        self._save_data(test_data, "test")

    def setup(self, stage: str):
        if stage == "fit":
            self._train_dataset = Dataset(self._prepared_data_dir / "train")
            self._val_dataset = Dataset(self._prepared_data_dir / "val")
        elif stage == "test":
            self._test_dataset = Dataset(self._prepared_data_dir / "test")
        else:
            raise ValueError(f"Unsupported stage: {stage}")

    def teardown(self, stage: str):
        if stage == "fit":
            del self._train_dataset
            del self._val_dataset
            rmtree(self._prepared_data_dir / "train")
            rmtree(self._prepared_data_dir / "val")
        elif stage == "test":
            del self._test_dataset
            rmtree(self._prepared_data_dir / "test")
        else:
            raise ValueError(f"Unsupported stage: {stage}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            collate_fn=Collator(),
            num_workers=self._num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            collate_fn=Collator(),
            num_workers=self._num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            collate_fn=Collator(),
            num_workers=self._num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def _save_data(self, data: list[tuple[str, int]], name: str):
        save_dir = self._prepared_data_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)
        for index, (description, label) in enumerate(data):
            output = {"description": description, "label": label}
            path = save_dir / f"{index}.json"
            path.write_text(json.dumps(output))


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
