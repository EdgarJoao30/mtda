# general
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

# deep learning
import torch
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning import Trainer, seed_everything

# custom modules
from py_module.data_module import Koumbia_DataModule
from py_module.task_module import ClassifyTimeSeries
#from py_module.writer import PredictionWriter

# helper class for accumulating best val accuracies
class AccBestVal(LambdaCallback):
    def __init__(self):
        self.evolution = []

    def on_validation_epoch_end(self, trainer, pl_module):
        self.evolution.append(deepcopy(trainer.callback_metrics))


# wrapper for training & evaluation
class NN:
    def __init__(self, **kwargs):
        self.split = kwargs.get("split")
        self.model = kwargs.get("model_name")
        self.batch_size = kwargs.get("batch_size")
        self.optimiser = kwargs.get("optimiser")
        self.loss = kwargs.get("loss")
        self.lr = kwargs.get("lr")
        self.max_epochs = kwargs.get("max_epochs")
        self.patience = kwargs.get("patience")
        self.seed = kwargs.get("seed")
        self.save_dir = kwargs.get("save_dir")
        self.wandb_project = kwargs.get("wandb_project")
        self.wandb_runname = kwargs.get("wandb_runname")
        self.wandb_notes = kwargs.get("wandb_notes")
        # create save folder if necessary
        self.save_path = Path(self.save_dir, self.wandb_runname)
        os.makedirs(self.save_path, exist_ok=True)
        seed_everything(self.seed, workers=True)
        # define wandb session & log model configuration
        configs = {}
        configs["py_script"] = os.path.basename(__file__)
        configs = {**configs, **kwargs}
        self.wandb_logger = WandbLogger(
            entity="mtda",
            save_dir=self.save_path,
            project=self.wandb_project,
            notes=self.wandb_notes,
        )
        wandb.run.name = self.wandb_runname
        wandb.config.update(configs)

    def run(self):
        # imports specific for the current run
        exec(f"from py_module.loss import {self.loss}")
        exec(f"from py_module.model import {self.model}")

        # fixed parameters for single GPU training
        
        accelerator = "gpu"
        
        gpus_per_node = 1
        num_nodes = 1
        #strategy = None
        #num_workers = 4

        # set up data loader
        dm = Koumbia_DataModule(self.batch_size, self.split)

        # set up model
        if self.model == 'TempCNN':
            model = eval(f"{self.model}(10, 8)")
        elif self.model == 'InceptionTime':
            model = eval(f"{self.model}(8, device=torch.device('cuda:0'))")
        
        criterion = eval(f"{self.loss}()")

        if self.optimiser == "SGD":
            optimiser = torch.optim.SGD(model.parameters(), lr=self.lr)
        elif self.optimiser == "Adam":
            optimiser = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-6, betas=[0.9, 0.999], eps=1e-8)
        elif self.optimiser == "Adam_default":
            optimiser = torch.optim.Adam(model.parameters(), lr=self.lr)

        # define eval interval of 1000 samples
        interval_n_iters = int(1000 / self.batch_size)

        scheduler = ReduceLROnPlateau(
            optimizer=optimiser,
            mode="min",
            factor=0.5,
            patience=int((self.patience / 4) * interval_n_iters),
            cooldown=4,
            min_lr=1e-7,
        )
        TS_module = ClassifyTimeSeries(
            model=model,
            num_classes=8,
            criterion=criterion,
            optimizer=optimiser,
            scheduler=scheduler
        )

        # define callbacks
        ckpt_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=os.path.join(self.save_path, "checkpoints"),
            filename="ckpt-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            mode="min",
            save_weights_only=True,
        )
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=self.patience,
            mode="min",
        )
        acc_best_callback = AccBestVal()
        prog_rate = TQDMProgressBar(refresh_rate=10)
        callbacks = [
            ckpt_callback,
            early_stop_callback,
            acc_best_callback,
            prog_rate,
        ]

        # instantiation & execution of training
        trainer = Trainer(
            accelerator=accelerator,
            devices=gpus_per_node,
            #strategy=strategy,
            num_nodes=num_nodes,
            max_epochs=self.max_epochs,
            num_sanity_val_steps=0,
            val_check_interval=interval_n_iters,
            precision='16-mixed',
            deterministic="warn",
            callbacks=callbacks,
            logger=self.wandb_logger,
            enable_progress_bar=True,
        )
        trainer.fit(TS_module, datamodule=dm)
        trainer.validate(TS_module, datamodule=dm)
        
        # logging best model's accuracy
        if early_stop_callback.state_dict()["stopped_epoch"] > 0:
            acc_evol = [d for d in acc_best_callback.evolution if d] 
            best_val_loss = early_stop_callback.state_dict()["best_score"]
            best_idx = [x['val_loss'] for x in acc_evol].index(best_val_loss)
            best_val_f1 = round(acc_evol[best_idx]["val_MulticlassF1Score"].item(), 4)
            best_val_acc = round(acc_evol[best_idx]["val_MulticlassAccuracy"].item(), 4)
            self.wandb_logger.experiment.summary["best_val_MulticlassF1Score"] = best_val_f1
            self.wandb_logger.experiment.summary["best_val_MulticlassAccuracy"] = best_val_acc

        # load best model
        # best_model = TS_module.load_from_checkpoint(
        #     ckpt_callback.best_model_path
        # )
        path_ckpt = os.listdir(os.path.join(self.save_path, "checkpoints"))[-1]
        path_ckpt = os.path.join(self.save_path, "checkpoints", path_ckpt)

        TS_module = ClassifyTimeSeries.load_from_checkpoint(
            path_ckpt,
            model=model,
            num_classes=8,
            criterion=criterion,
            optimizer=optimiser
        )        
        # test best model
        trainer.test(TS_module, datamodule=dm)
        
        
        # # predicting based on best model
        # writer_callback = PredictionWriter(
        #     output_dir=os.path.join(self.save_path, "predictions"),
        #     write_interval="batch",
        # )
        # trainer = Trainer(
        #     accelerator=accelerator,
        #     devices=gpus_per_node,
        #     strategy=strategy,
        #     num_nodes=num_nodes,
        #     #callbacks=[writer_callback],
        #     enable_progress_bar=True,
        # )
        # trainer.predict(TS_module, datamodule=dm)

if __name__ == "__main__":

    # parse arguments
    import argparse
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(
        description="Train & evaluate a Time Series classification model",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--split",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="TempCNN",
        help="see model.py for avaliable models",
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16
    )
    parser.add_argument(
        "--optimiser",
        type=str,
        choices=["SGD", "Adam", 'Adam_default'],
        default="Adam",
    )

    parser.add_argument(
        "--loss",
        type=str,
        default="ce_balanced",
        help="see loss.py for avaliable losses",
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-3
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--patience", 
        type=int, 
        default=100
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/edgar/DATA/results/models/",
    )
    parser.add_argument(
        "--wandb_project", 
        type=str, 
        default="tvt"
    )
    parser.add_argument(
        "--wandb_runname",
        type=str,
        default=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
    )
    parser.add_argument(
        "--wandb_notes",
        type=str,
        default=None,
        help="additional notes/remarks for this run",
    )

    config = vars(parser.parse_args())

    # train & evaluate model
    seg_trainer = NN(**config)
    seg_trainer.run()
