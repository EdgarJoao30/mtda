import torch
from torchmetrics import MeanMetric, MetricCollection, ClasswiseWrapper
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
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
            self.train_epoch_loss, self.val_epoch_loss = None, None
            self.train_epoch_metrics, self.val_epoch_metrics, self.val_epoch_cw_metrics = None, None, None

            self.train_metrics = MetricCollection([MulticlassF1Score(num_classes=self.num_classes,average='weighted'),
                                                   MulticlassAccuracy(num_classes=self.num_classes, average='weighted')])
            self.val_metrics = MetricCollection([MulticlassF1Score(num_classes=self.num_classes,average='weighted'),
                                                   MulticlassAccuracy(num_classes=self.num_classes, average='weighted')])
            
            self.val_cw_metrics= MetricCollection({'F1Score': ClasswiseWrapper(MulticlassF1Score(num_classes=self.num_classes,average='none')),
                                                   'Accuracy': ClasswiseWrapper(MulticlassAccuracy(num_classes=self.num_classes, average='none'))})
            
            self.train_loss = MeanMetric()
            self.val_loss = MeanMetric()

        elif stage == "validate":
            self.val_epoch_loss, self.val_epoch_metrics, self.val_epoch_cw_metrics  = None, None, None
            self.val_metrics = MetricCollection([MulticlassF1Score(num_classes=self.num_classes,average='weighted'),
                                                   MulticlassAccuracy(num_classes=self.num_classes, average='weighted')])
            self.val_cw_metrics= MetricCollection({'F1Score': ClasswiseWrapper(MulticlassF1Score(num_classes=self.num_classes,average='none')),
                                                   'Accuracy': ClasswiseWrapper(MulticlassAccuracy(num_classes=self.num_classes, average='none'))})
            
            self.val_loss = MeanMetric()
            
        elif stage == "test":
            self.test2018_epoch_metrics = None
            self.test2020_epoch_metrics = None
            self.test2021_epoch_metrics = None
            
            self.test2018_epoch_cw_metrics = None
            self.test2020_epoch_cw_metrics = None
            self.test2021_epoch_cw_metrics = None
            
            self.preds2018 = []
            self.y2018 = []
            
            self.preds2020 = []
            self.y2020 = []
            
            self.preds2021 = []
            self.y2021 = []
            
            self.test2018_metrics = MetricCollection([MulticlassF1Score(num_classes=self.num_classes,average='weighted'),
                                                   MulticlassAccuracy(num_classes=self.num_classes, average='weighted')])
            self.test2020_metrics = MetricCollection([MulticlassF1Score(num_classes=self.num_classes,average='weighted'),
                                                   MulticlassAccuracy(num_classes=self.num_classes, average='weighted')])
            self.test2021_metrics= MetricCollection([MulticlassF1Score(num_classes=self.num_classes,average='weighted'),
                                                   MulticlassAccuracy(num_classes=self.num_classes, average='weighted')])

            self.test2018_cw_metrics= MetricCollection({'F1Score': ClasswiseWrapper(MulticlassF1Score(num_classes=self.num_classes,average='none')),
                                                   'Accuracy': ClasswiseWrapper(MulticlassAccuracy(num_classes=self.num_classes, average='none'))})
            
            self.test2020_cw_metrics= MetricCollection({'F1Score': ClasswiseWrapper(MulticlassF1Score(num_classes=self.num_classes,average='none')),
                                                   'Accuracy': ClasswiseWrapper(MulticlassAccuracy(num_classes=self.num_classes, average='none'))})
            self.test2021_cw_metrics= MetricCollection({'F1Score': ClasswiseWrapper(MulticlassF1Score(num_classes=self.num_classes,average='none')),
                                                   'Accuracy': ClasswiseWrapper(MulticlassAccuracy(num_classes=self.num_classes, average='none'))})


    def forward(self, s2, s1):
        fused_logits, s2_logits, s1_logits = self.model(s2, s1)
        return fused_logits, s2_logits, s1_logits

    def step(self, batch):
        s2, s1, targets = batch[0], batch[1], batch[2] -1
        fused_logits, s2_logits, s1_logits = self.forward(s2, s1)
        
        fused_proba = torch.softmax(fused_logits, dim=1)
        
        fused_loss = self.criterion(fused_logits, targets)
        s2_loss = self.criterion(s2_logits, targets)
        s1_loss = self.criterion(s1_logits, targets)
        
        combined_loss = fused_loss + (0.3*s2_loss) + (0.3*s1_loss)
        with torch.no_grad():
            preds = torch.argmax(fused_proba, dim=1) 
        return combined_loss, preds, targets

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        
        self.train_loss.update(loss)
        self.train_metrics(preds, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        self.val_loss.update(loss)
        self.val_metrics(preds, targets)
        self.val_cw_metrics(preds, targets)
        return loss
    
    def test_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            loss, preds, targets = self.step(batch)
            self.preds2018.append(preds)
            self.y2018.append(targets)
            self.test2018_metrics(preds, targets)
            self.test2018_cw_metrics(preds, targets)
            return loss
        elif dataloader_idx ==1:
            loss, preds, targets = self.step(batch)
            self.preds2020.append(preds)
            self.y2020.append(targets)
            self.test2020_metrics(preds, targets)
            self.test2020_cw_metrics(preds, targets)
            return loss
        elif dataloader_idx ==2:
            loss, preds, targets = self.step(batch)
            self.preds2021.append(preds)
            self.y2021.append(targets)
            self.test2021_metrics(preds, targets)
            self.test2021_cw_metrics(preds, targets)
            return loss
        
    def on_train_epoch_end(self):
        self.train_epoch_loss = self.train_loss.compute()
        self.train_epoch_metrics = self.train_metrics.compute()
        self.log("train_loss",self.train_epoch_loss,on_step=False,on_epoch=True,prog_bar=True,logger=True,rank_zero_only=True,)
        self.train_loss.reset()
        self.train_metrics.reset()
        
    def on_validation_epoch_end(self):
        self.val_epoch_loss = self.val_loss.compute()
        self.val_epoch_metrics = self.val_metrics.compute()
        self.log("val_loss",self.val_epoch_loss,on_step=False,on_epoch=True,prog_bar=True,logger=True,rank_zero_only=True,)
        self.log("val_MulticlassF1Score",self.val_epoch_metrics['MulticlassF1Score'],on_step=False,on_epoch=True,prog_bar=True,logger=True,rank_zero_only=True,)
        self.log("val_MulticlassAccuracy",self.val_epoch_metrics['MulticlassAccuracy'],on_step=False,on_epoch=True,prog_bar=True,logger=True,rank_zero_only=True,)
        self.val_loss.reset()
        self.val_metrics.reset()
        
        self.val_epoch_cw_metrics = self.val_cw_metrics.compute()
        for i in range(self.num_classes):
                self.log(f"val_F1Score_{i}",self.val_epoch_cw_metrics[f'multiclassf1score_{i}'],on_step=False,on_epoch=True,prog_bar=True,logger=True,rank_zero_only=True,)
                self.log(f"val_Accuracy_{i}",self.val_epoch_cw_metrics[f'multiclassaccuracy_{i}'],on_step=False,on_epoch=True,prog_bar=True,logger=True,rank_zero_only=True,)
        self.val_cw_metrics.reset()
        
    def on_test_epoch_end(self):
        for year, epoch_metric, metric_year in zip([2018, 2020, 2021], [self.test2018_epoch_metrics, self.test2020_epoch_metrics, self.test2021_epoch_metrics], [self.test2018_metrics, self.test2020_metrics, self.test2021_metrics]):
            epoch_metric = metric_year.compute()
            self.log(f"test{year}_MulticlassF1Score", epoch_metric['MulticlassF1Score'],on_step=False,on_epoch=True,prog_bar=True,logger=True,rank_zero_only=True,)
            self.log(f"test{year}_MulticlassAccuracy", epoch_metric['MulticlassAccuracy'],on_step=False,on_epoch=True,prog_bar=True,logger=True,rank_zero_only=True,)
            metric_year.reset()
            
        for year, epoch_metric, metric_year in zip([2018, 2020, 2021], [self.test2018_epoch_cw_metrics, self.test2020_epoch_cw_metrics, self.test2021_epoch_cw_metrics], [self.test2018_cw_metrics, self.test2020_cw_metrics, self.test2021_cw_metrics]):
            epoch_metric = metric_year.compute()
            for i in range(self.num_classes):
                self.log(f"test{year}_F1Score_{i}",epoch_metric[f'multiclassf1score_{i}'],on_step=False,on_epoch=True,prog_bar=True,logger=True,rank_zero_only=True,)
                self.log(f"test{year}_Accuracy_{i}",epoch_metric[f'multiclassaccuracy_{i}'],on_step=False,on_epoch=True,prog_bar=True,logger=True,rank_zero_only=True,)
            metric_year.reset()  


    # def predict_step(self, batch):
    #     logits = self.forward(batch["X"])
    #     proba = torch.softmax(logits, dim=1)
    #     batch["preds"] = torch.argmax(proba, dim=1)
    #     return batch

    def configure_optimizers(self):

        return self.optimizer
