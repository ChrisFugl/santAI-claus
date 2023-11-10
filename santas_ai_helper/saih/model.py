import lightning.pytorch as pl
import torch
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
        return outputs.loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)
        return optimizer
