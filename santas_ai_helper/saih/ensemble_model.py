import lightning.pytorch as pl
import torch
import torchmetrics

from saih.model import Model


class EnsembleModel(pl.LightningModule):
    def __init__(self, model_dirs: list[str]):
        super().__init__()

        self._models = torch.nn.ModuleList()
        for model_dir in model_dirs:
            model = Model.load_from_checkpoint(model_dir)
            self._models.append(model)

        self._accuracy = torchmetrics.Accuracy(task="binary")
        self._precision = torchmetrics.Precision(task="binary")
        self._recall = torchmetrics.Recall(task="binary")
        self._f1 = torchmetrics.F1Score(task="binary")

    def test_step(self, batch, batch_idx):
        inputs, labels = batch

        batch_size = labels.shape[0]
        logits = torch.zeros([batch_size, 2], dtype=torch.float32, device=self.device)
        for model in self._models:
            outputs = model(
                **inputs,
                labels=labels,
                output_hidden_states=False,
                output_attentions=False,
            )
            logits += outputs.logits

        # TODO: perhaps average the logits instead of summing them?

        self.log("loss", outputs.loss, on_step=False, on_epoch=True)

        predictions = torch.argmax(outputs.logits, dim=1)

        self._accuracy(predictions, labels)
        self._precision(predictions, labels)
        self._recall(predictions, labels)
        self._f1(predictions, labels)

        self.log("accuracy", self._accuracy, on_step=False, on_epoch=True)
        self.log("precision", self._precision, on_step=False, on_epoch=True)
        self.log("recall", self._recall, on_step=False, on_epoch=True)
        self.log("f1", self._f1, on_step=False, on_epoch=True)

        return outputs.loss
