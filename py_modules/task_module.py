import torch
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, JaccardIndex
import pytorch_lightning as pl


class ClassifyTimeSeries(pl.LightningModule):
    def __init__(
        self,
        model,
        num_classes,
        criterion,
        optimizer,
        scheduler=None,
    ):

        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def setup(self, stage=None):
        if stage == "fit":
            self.train_epoch_loss, self.val_epoch_loss, self.test_epoch_loss = None, None, None
            self.train_epoch_metrics, self.val_epoch_metrics, self.test_epoch_metrics = None, None, None

            self.train_metrics = MetricCollection([MulticlassF1Score(num_classes=self.num_classes,average='weighted'),
                                                   MulticlassAccuracy(num_classes=self.num_classes, average='weighted')])
            self.val_metrics = MetricCollection([MulticlassF1Score(num_classes=self.num_classes,average='weighted'),
                                                   MulticlassAccuracy(num_classes=self.num_classes, average='weighted')])
            self.train_loss = MeanMetric()
            self.val_loss = MeanMetric()

        elif stage == "validate":
            self.val_epoch_loss, self.val_epoch_metrics = None, None
            self.val_metrics = MetricCollection([MulticlassF1Score(num_classes=self.num_classes,average='weighted'),
                                                   MulticlassAccuracy(num_classes=self.num_classes, average='weighted')])
            self.val_loss = MeanMetric()
            
        elif stage == "test":
            self.test_epoch_loss, self.test_epoch_metrics = None, None
            self.test_metrics = MetricCollection([MulticlassF1Score(num_classes=self.num_classes,average='weighted'),
                                                   MulticlassAccuracy(num_classes=self.num_classes, average='weighted')])
            self.test_loss = MeanMetric()

    def forward(self, input_im):
        logits = self.model(input_im)
        return logits

    def step(self, batch):
        X, targets = batch[0], batch[1] -1
        logits = self.forward(X)
        proba = torch.softmax(logits, dim=1)
        loss = self.criterion(logits, targets)
        with torch.no_grad():
            preds = torch.argmax(proba, dim=1) 
        return loss, preds, targets

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        
        self.train_loss.update(loss)
        self.train_metrics(preds, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        self.val_loss.update(loss)
        self.val_metrics(preds, targets)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        self.test_loss.update(loss)
        self.test_metrics(preds, targets)
        return loss
    
    def on_train_epoch_end(self):
        self.train_epoch_loss = self.train_loss.compute()
        self.train_epoch_metrics = self.train_metrics.compute()
        self.log(
            "train_epoch_loss",
            self.train_epoch_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
        )
        self.train_loss.reset()
        self.train_metrics.reset()
        
    def on_validation_epoch_end(self):
        self.val_epoch_loss = self.val_loss.compute()
        self.val_epoch_metrics = self.val_metrics.compute()
        self.log(
            "val_loss",
            self.val_epoch_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
        )
        self.log(
            "val_MulticlassF1Score",
            self.val_epoch_metrics['MulticlassF1Score'],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
        )
        self.log(
            "val_MulticlassAccuracy",
            self.val_epoch_metrics['MulticlassAccuracy'],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
        )
        self.val_loss.reset()
        self.val_metrics.reset()
        
    def on_test_epoch_end(self):
        self.test_epoch_loss = self.test_loss.compute()
        self.test_epoch_metrics = self.test_metrics.compute()
        self.log(
            "test_loss",
            self.test_epoch_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
        )
        self.log(
            "test_MulticlassF1Score",
            self.test_epoch_metrics['MulticlassF1Score'],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
        )
        self.log(
            "test_MulticlassAccuracy",
            self.test_epoch_metrics['MulticlassAccuracy'],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
        )
        self.test_loss.reset()
        self.test_metrics.reset()        


    # def predict_step(self, batch):
    #     logits = self.forward(batch["X"])
    #     proba = torch.softmax(logits, dim=1)
    #     batch["preds"] = torch.argmax(proba, dim=1)
    #     return batch

    def configure_optimizers(self):
        if self.scheduler is not None:
            lr_scheduler_config = {
                "scheduler": self.scheduler,
                "interval": "step",
                "monitor": "val_loss",
                "frequency": 1,
                "strict": False,
                "name": "Scheduler",
            }
            config = {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler_config}
            return config
        else:
            return self.optimizer