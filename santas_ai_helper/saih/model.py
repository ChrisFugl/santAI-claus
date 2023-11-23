import lightning.pytorch as pl
import torch
import torchmetrics
from transformers import BertForSequenceClassification

from saih.constants import LABEL_TO_CLASS
from saih.constants import MODEL_NAME


class Model(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float,
    ):
        super().__init__()

        self._learning_rate = learning_rate

        self._model = BertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=len(LABEL_TO_CLASS),
        )

        self._accuracy = torchmetrics.Accuracy(task="binary")
        self._precision = torchmetrics.Precision(task="binary")
        self._recall = torchmetrics.Recall(task="binary")
        self._f1 = torchmetrics.F1Score(task="binary")

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self._model(
            **inputs,
            labels=labels,
            output_hidden_states=False,
            output_attentions=False,
        )
        self.log("train/loss", outputs.loss, on_step=True, on_epoch=True, prog_bar=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self._model(
            **inputs,
            labels=labels,
            output_hidden_states=False,
            output_attentions=False,
        )
        self.log("val/loss", outputs.loss, on_step=False, on_epoch=True)

        predictions = torch.argmax(outputs.logits, dim=1)

        self._accuracy(predictions, labels)
        self._precision(predictions, labels)
        self._recall(predictions, labels)
        self._f1(predictions, labels)

        self.log("val/accuracy", self._accuracy, on_step=False, on_epoch=True)
        self.log("val/precision", self._precision, on_step=False, on_epoch=True)
        self.log("val/recall", self._recall, on_step=False, on_epoch=True)
        self.log("val/f1", self._f1, on_step=False, on_epoch=True)

        return outputs.loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self._model(
            **inputs,
            labels=labels,
            output_hidden_states=False,
            output_attentions=False,
        )
        self.log("test/loss", outputs.loss, on_step=False, on_epoch=True)

        predictions = torch.argmax(outputs.logits, dim=1)

        self._accuracy(predictions, labels)
        self._precision(predictions, labels)
        self._recall(predictions, labels)
        self._f1(predictions, labels)

        self.log("val/accuracy", self._accuracy, on_step=False, on_epoch=True)
        self.log("val/precision", self._precision, on_step=False, on_epoch=True)
        self.log("val/recall", self._recall, on_step=False, on_epoch=True)
        self.log("val/f1", self._f1, on_step=False, on_epoch=True)

        return outputs.loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)
        return optimizer
