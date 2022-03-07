from pathlib import Path
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import (
    AdamW,
    Optimizer,
)
from torchmetrics.functional import (
    accuracy,
    f1_score,
    precision_recall
)
from transformers import BertForSequenceClassification


class BertClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        lr: float = 1e-5,
        num_labels: int = 2,
    ) -> None:
        super().__init__()

        self._model = BertForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
        self._lr = lr

    def configure_optimizers(
        self,
    ) -> Optimizer:
        optimizer = AdamW(
            self._model.parameters(),
            lr=self._lr,
            amsgrad=True,
        )

        return optimizer

    def forward(
        self,
        x,
    ):
        return self._model(**x)

    def training_step(self, batch, batch_idx):
        labels = batch.pop('Class')
        logits = self._model(**batch).logits
        loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(axis=1)

        self.log('train/loss', loss.item())
        self._calculate_metrics(
            predictions,
            labels,
            'train',
        )

        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch.pop('Class')
        logits = self._model(**batch).logits
        loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(axis=1)

        self.log('val/loss', loss.item())
        self._calculate_metrics(
            predictions,
            labels,
            'val',
        )

    def _calculate_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        prefix: str,
    ) -> None:
        self.log(
            f'{prefix}/accuracy',
            accuracy(
                predictions,
                labels,
            ),
        )
        self.log(
            f'{prefix}/f1',
            f1_score(
                predictions,
                labels,
            ),
        )

        prec, rec = precision_recall(
            predictions,
            labels,
        )
        self.log(
            f'{prefix}/precision',
            prec,
        )
        self.log(
            f'{prefix}/recall',
            rec,
        )
