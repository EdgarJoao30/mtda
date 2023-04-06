import torch
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, JaccardIndex
import pytorch_lightning as pl
import numpy as np
import pdb


class ClassifyTimeSeries(pl.LightningModule):
    def __init__(
        self,
        model,
        num_classes,
        criterion,
        criterion_pseudo,
        optimizer,
        beta,
        max_epochs,
        scheduler=None,
    ):

        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.criterion = criterion
        self.criterion_pseudo = criterion_pseudo
        self.optimizer = optimizer
        self.beta = beta
        self.max_epochs = max_epochs
        self.scheduler = scheduler

    def setup(self, stage=None):
        if stage == "fit":
            self.train_epoch_in_loss= None
            self.train_epoch_domain_loss = None
            self.train_epoch_pseudo_loss = None
            self.train_epoch_combined_loss = None
            
            self.train_epoch_metrics = None

            self.train_metrics = MetricCollection([MulticlassF1Score(num_classes=self.num_classes,average='weighted'),
                                                   MulticlassAccuracy(num_classes=self.num_classes, average='weighted')])

            self.train_in_loss = MeanMetric()
            self.train_domain_loss = MeanMetric()
            self.train_pseudo_loss = MeanMetric()
            self.train_combined_loss = MeanMetric()
            
        elif stage == "test":
            self.test2018_epoch_metrics = None
            self.test2020_epoch_metrics = None
            self.test2021_epoch_metrics = None
            self.test2018_metrics = MetricCollection([MulticlassF1Score(num_classes=self.num_classes,average='weighted'),
                                                   MulticlassAccuracy(num_classes=self.num_classes, average='weighted')])
            self.test2020_metrics = MetricCollection([MulticlassF1Score(num_classes=self.num_classes,average='weighted'),
                                                   MulticlassAccuracy(num_classes=self.num_classes, average='weighted')])
            self.test2021_metrics= MetricCollection([MulticlassF1Score(num_classes=self.num_classes,average='weighted'),
                                                   MulticlassAccuracy(num_classes=self.num_classes, average='weighted')])


    def forward(self, input_im, lamb_da = 1.0):
        logits = self.model(input_im, lamb_da)
        return logits

    def training_step(self, batch, batch_idx):
        alpha = self.beta * (float(self.current_epoch) / self.max_epochs)
        lamb_da = 2 / (1 + np.exp(-10 * (float(self.current_epoch) / self.max_epochs), dtype=np.float32)) - 1
        lamb_da = lamb_da.astype(np.float32)
        # load batches
        source_X, source_y, target_X = batch[0], batch[1] -1, batch[2]
        # forward pass
        _, source_logits_class, source_logits_dom = self.forward(source_X, lamb_da)
        _, out_logits_class, out_logits_dom = self.forward(target_X, lamb_da)
        # Create domain labels
        domain_y = torch.cat([torch.ones(batch[0].shape[0]), torch.zeros(batch[2].shape[0])])
        # Probabilities
        source_proba_class = torch.softmax(source_logits_class, dim=1)
        target_proba_class = torch.softmax(out_logits_class, dim=1)
        # Predictions
        with torch.no_grad():
            source_pred = torch.argmax(source_proba_class, axis=1)
            target_pred = torch.argmax(target_proba_class, axis=1)
            # Select and calculate loss on pseudo-labels
            first_cond = (source_pred == source_y)
            second_cond = (source_pred == target_pred)
            result = (first_cond & second_cond).type(torch.int)

        # Calculate loss on label prediction on source domain
        loss_class = self.criterion(source_logits_class, source_y)
        # Calculate loss on domain prediction
        loss_dom = self.criterion(torch.cat([source_logits_dom, out_logits_dom]), domain_y.type(torch.long).to(self.device))
        # Calculation of loss on pseudo-labels on source domain
        loss_pseudo = self.criterion_pseudo(out_logits_class, torch.tensor(target_pred))
        loss_pseudo = torch.mean(loss_pseudo * torch.tensor(result).to(self.device))
        # Calculation of combined loss
        loss_combined = (1 - alpha) * (loss_class + loss_dom) + alpha * loss_pseudo
        
        self.train_in_loss.update(loss_class)
        self.train_domain_loss.update(loss_dom)
        self.train_pseudo_loss.update(loss_pseudo)
        self.train_combined_loss.update(loss_combined)
        
        self.train_metrics.update(source_pred, source_y)
        
        return loss_combined
    
    def test_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            X, y = batch[0], batch[1] -1
            _, logits_class, __ = self.forward(X)
            proba_class = torch.softmax(logits_class, dim=1)
            pred = torch.argmax(proba_class, axis=1)
            loss = self.criterion(logits_class, y)
            self.test2018_metrics(pred, y)
            return loss
        elif dataloader_idx ==1:
            X, y = batch[0], batch[1] -1
            _, logits_class, __ = self.forward(X)
            proba_class = torch.softmax(logits_class, dim=1)
            pred = torch.argmax(proba_class, axis=1)
            loss = self.criterion(logits_class, y)
            self.test2020_metrics(pred, y)
            return loss
        elif dataloader_idx ==2:
            X, y = batch[0], batch[1] -1
            _, logits_class, __ = self.forward(X)
            proba_class = torch.softmax(logits_class, dim=1)
            pred = torch.argmax(proba_class, axis=1)
            loss = self.criterion(logits_class, y)
            self.test2021_metrics(pred, y)
            return loss
        
    def on_train_epoch_end(self):
        self.train_epoch_in_loss = self.train_in_loss.compute()
        self.train_epoch_domain_loss = self.train_domain_loss.compute()
        self.train_epoch_pseudo_loss = self.train_pseudo_loss.compute()
        self.train_epoch_combined_loss = self.train_combined_loss.compute()
        
        self.train_epoch_metrics = self.train_metrics.compute()
        self.log("train_in_loss",self.train_epoch_in_loss,on_step=False,on_epoch=True,prog_bar=True,logger=True,rank_zero_only=True,)
        self.train_in_loss.reset()
        
        self.log("train_domain_loss",self.train_epoch_domain_loss,on_step=False,on_epoch=True,prog_bar=True,logger=True,rank_zero_only=True,)
        self.train_domain_loss.reset()
        
        self.log("train_pseudo_loss",self.train_epoch_pseudo_loss,on_step=False,on_epoch=True,prog_bar=True,logger=True,rank_zero_only=True,)
        self.train_pseudo_loss.reset()
        
        self.log("train_combined_loss",self.train_epoch_combined_loss,on_step=False,on_epoch=True,prog_bar=True,logger=True,rank_zero_only=True,)
        self.train_combined_loss.reset()
        
        self.log("MulticlassF1Score",self.train_epoch_metrics['MulticlassF1Score'],on_step=False,on_epoch=True,prog_bar=True,logger=True,rank_zero_only=True,)
        self.log("MulticlassAccuracy",self.train_epoch_metrics['MulticlassAccuracy'],on_step=False,on_epoch=True,prog_bar=True,logger=True,rank_zero_only=True,)
        self.train_metrics.reset()
        
    def on_test_epoch_end(self):
        self.test_epoch_metrics = self.test2018_metrics.compute()
        self.log("test2018_MulticlassF1Score",self.test_epoch_metrics['MulticlassF1Score'],on_step=False,on_epoch=True,prog_bar=True,logger=True,rank_zero_only=True,)
        self.log( "test2018_MulticlassAccuracy",self.test_epoch_metrics['MulticlassAccuracy'],on_step=False,on_epoch=True,prog_bar=True,logger=True,rank_zero_only=True,)
        self.test2018_metrics.reset()      
        
        self.test_epoch_metrics = self.test2020_metrics.compute()
        self.log("test2020_MulticlassF1Score",self.test_epoch_metrics['MulticlassF1Score'],on_step=False,on_epoch=True,prog_bar=True,logger=True,rank_zero_only=True,)
        self.log( "test2020_MulticlassAccuracy",self.test_epoch_metrics['MulticlassAccuracy'],on_step=False,on_epoch=True,prog_bar=True,logger=True,rank_zero_only=True,)
        self.test2020_metrics.reset()  
        
        self.test_epoch_metrics = self.test2021_metrics.compute()
        self.log("test2021_MulticlassF1Score",self.test_epoch_metrics['MulticlassF1Score'],on_step=False,on_epoch=True,prog_bar=True,logger=True,rank_zero_only=True,)
        self.log( "test2021_MulticlassAccuracy",self.test_epoch_metrics['MulticlassAccuracy'],on_step=False,on_epoch=True,prog_bar=True,logger=True,rank_zero_only=True,)
        self.test2021_metrics.reset()    


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