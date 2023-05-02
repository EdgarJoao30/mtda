# general
import os
from datetime import datetime
from pathlib import Path

# deep learning
import torch
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything

# custom modules
from py_module.data_module_mm import Koumbia_DataModule
from py_module.task_module_mm import ClassifyTimeSeries
from py_module.callbacks import get_callbacks
#from py_module.writer import PredictionWriter


# wrapper for training & evaluation
class NN:
    def __init__(self, **kwargs):
        self.split = kwargs.get("split")
        self.in_domain = kwargs.get("in_domain")
        self.model = kwargs.get("model_name")
        self.batch_size = kwargs.get("batch_size")
        self.optimiser = kwargs.get("optimiser")
        self.loss = kwargs.get("loss")
        self.lr = kwargs.get("lr")
        self.max_epochs = kwargs.get("max_epochs")
        # self.patience = kwargs.get("patience")
        self.seed = kwargs.get("seed")
        self.save_dir = kwargs.get("save_dir")
        self.wandb_project = kwargs.get("wandb_project")
        self.wandb_runname = kwargs.get("wandb_runname")
        self.wandb_notes = kwargs.get("wandb_notes")
        # create save folder if necessary
        self.save_path = Path(self.save_dir, self.wandb_runname)
        os.makedirs(self.save_path, exist_ok=True)
        seed_everything(self.seed, workers=True)
        #callbacks
        self.callbacks = get_callbacks(self.save_path)
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
        dm = Koumbia_DataModule(self.batch_size, self.split, self.in_domain)

        # set up model
        model = eval(f"{self.model}(8)")
        criterion = eval(f"{self.loss}()")

        if self.optimiser == "SGD":
            optimiser = torch.optim.SGD(model.parameters(), lr=self.lr)
        elif self.optimiser == "Adam":
            optimiser = torch.optim.Adam(model.parameters(), lr=self.lr) # weight_decay=1e-6

        TS_module = ClassifyTimeSeries(
            model=model,
            num_classes=8,
            criterion=criterion,
            optimizer=optimiser,
        )

        # instantiation & execution of training
        trainer = Trainer(
            accelerator=accelerator,
            devices=gpus_per_node,
            num_nodes=num_nodes,
            max_epochs=self.max_epochs,
            num_sanity_val_steps=0,
            precision='16-mixed',
            deterministic="warn",
            callbacks=self.callbacks,
            logger=self.wandb_logger,
            enable_progress_bar=True,
            # fast_dev_run=True
        )
        trainer.fit(TS_module, datamodule=dm)
        trainer.validate(TS_module, datamodule=dm)
        
        # logging best model's F1 score
        
        best_val_f1 = self.callbacks[0].state_dict()["best_model_score"]
        self.wandb_logger.experiment.summary["best_val_MulticlassF1Score"] = best_val_f1

        # load best model & test
        trainer.test(TS_module, datamodule=dm)
        
        # confusion matrix
        wandb.log({"conf_mat_2018" : wandb.plot.confusion_matrix(probs=None,
                        y_true= self.callbacks[1].y2018, preds= self.callbacks[1].preds2018)})
        wandb.log({"conf_mat_2020" : wandb.plot.confusion_matrix(probs=None,
                        y_true= self.callbacks[1].y2020, preds= self.callbacks[1].preds2020)})
        wandb.log({"conf_mat_2021" : wandb.plot.confusion_matrix(probs=None,
                        y_true= self.callbacks[1].y2021, preds= self.callbacks[1].preds2021)})
        
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
        "--in_domain",
        type=int,
        default=2018,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mmTempCNN",
        help="see model.py for avaliable models",
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=64
    )
    parser.add_argument(
        "--optimiser",
        type=str,
        choices=["SGD", "Adam"],
        default="Adam",
    )

    parser.add_argument(
        "--loss",
        type=str,
        default="ce",
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
        default=400,
    )
    # parser.add_argument(
    #     "--patience", 
    #     type=int, 
    #     default=100
    # )
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
        default="test_mm_v1"
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

    # train, evaluate and test model
    TS_trainer = NN(**config)
    TS_trainer.run()
