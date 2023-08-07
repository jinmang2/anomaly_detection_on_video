from typing import Dict, Any

import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve

import torch
import lightning.pytorch as pl

from .dataset import build_feature_dataset


class VideoAnomalyDetectionRunner(pl.LightningModule):
    # lightning.pytorch.core.module.LightningModule
    def __init__(self, model, optimizer, data):
        super().__init__()
        # `model` is an instance of `nn.Module` and is already saved during checkpointing.
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self.validation_step_outputs = []

    # lightning.pytorch.core.module.LightningModule
    def training_step(self, batch, batch_idx):
        ninputs, ainputs = batch
        inputs = torch.cat((ninputs["feature"], ainputs["feature"]), dim=0)
        nlabels = ninputs["anomaly"]
        alabels = ainputs["anomaly"]
        outputs = self.model(video=inputs, abnormal_labels=alabels, normal_labels=nlabels)
        log_kwargs = dict(prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("train_loss", outputs.loss, **log_kwargs)
        return outputs.loss

    # lightning.pytorch.core.module.LightningModule
    def validation_step(self, batch, batch_idx):
        features = batch["feature"].permute(0, 2, 1, 3)
        outputs = self.model(video=features)
        results = {
            "preds": outputs.scores.squeeze(0).squeeze(-1).cpu().detach().numpy(),
            "labels": batch["label"].squeeze(0).cpu().numpy(),
        }
        self.validation_step_outputs.append(results)
        return None

    # lightning.pytorch.core.module.LightningModule
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.optimizer.learning_rate,
            weight_decay=self.hparams.optimizer.weight_decay,
        )
        return [optimizer]

    # lightning.pytorch.core.hooks.ModelHooks
    def on_validation_epoch_end(self) -> None:
        """Called in the validation loop at the very end of the epoch."""
        outputs = self.validation_step_outputs

        all_preds = [output["preds"] for output in outputs]
        all_preds = np.concatenate(all_preds)
        all_preds = np.repeat(all_preds, self.hparams.data.frames_per_clip)

        all_labels = [output["labels"] for output in outputs]
        all_labels = np.concatenate(all_labels).tolist()

        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        rec_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        pr_auc = auc(recall, precision)
        self.log("valid/rec_auc", rec_auc, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("valid/pr_auc", pr_auc, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        self.validation_step_outputs.clear()

    # lightning.pytorch.core.hooks.DataHooks
    def setup(self, stage: str) -> None:
        self.train_dataset = build_feature_dataset(
            mode="train",
            revision=self.hparams.data.revision,
            cache_dir=self.hparams.data.cache_dir,
            dynamic_load=self.hparams.data.dynamic_load,
        )
        self.valid_dataset = build_feature_dataset(
            mode="test",
            revision=self.hparams.data.revision,
            cache_dir=self.hparams.data.cache_dir,
            dynamic_load=self.hparams.data.dynamic_load,
        )

    # lightning.pytorch.core.hooks.DataHooks
    def train_dataloader(self):
        return (
            torch.utils.data.DataLoader(
                self.train_dataset["normal"],
                batch_size=self.hparams.data.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=self.hparams.data.num_workers,
            ),
            torch.utils.data.DataLoader(
                self.train_dataset["abnormal"],
                batch_size=self.hparams.data.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=self.hparams.data.num_workers,
            ),
        )

    # lightning.pytorch.core.hooks.DataHooks
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=1,
            shuffle=False,
        )

    # lightning.pytorch.core.hooks.CheckpointHooks
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        pass

    # lightning.pytorch.core.hooks.CheckpointHooks
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        pass