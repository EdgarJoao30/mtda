from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from copy import deepcopy
import os
import numpy as np
import torch

class AccPseudoLabels(LambdaCallback):
    def __init__(self):
        self.pseudo_evolution_figure = []
        self.pseudo_evolution_table = []
        self.triggered_train = []
        self.triggered_test = []
        self.preds2018 = None
        self.preds2020 = None
        self.preds2021 = None
        self.y2018 = None
        self.y2020 = None
        self.y2021 = None
        
    def on_test_epoch_end(self, trainer, pl_module):
        
        self.triggered_test = 1
        self.preds2018 = np.array(torch.cat(pl_module.preds2018).to(device = 'cpu'))
        self.preds2020 = np.array(torch.cat(pl_module.preds2020).to(device = 'cpu'))
        self.preds2021 = np.array(torch.cat(pl_module.preds2021).to(device = 'cpu'))
        self.y2018 = np.array(torch.cat(pl_module.y2018).to(device = 'cpu'))
        self.y2020 = np.array(torch.cat(pl_module.y2020).to(device = 'cpu'))
        self.y2021 = np.array(torch.cat(pl_module.y2021).to(device = 'cpu'))

def get_callbacks(path = None):
    if path:
        save_path = os.path.join(path, "checkpoints")
    else:
        save_path = None
    ckpt_callback = ModelCheckpoint(
                monitor="val_MulticlassF1Score",
                dirpath=save_path,
                filename="ckpt-{epoch:02d}-{val_MulticlassF1Score:.2f}",
                save_top_k=1,
                mode="max",
                save_weights_only=True,
            )

    acc_pseudoLabels = AccPseudoLabels()

    prog_rate = TQDMProgressBar(refresh_rate=10)

    # early_stop_callback = EarlyStopping(
    #     monitor="val_loss",
    #     min_delta=0.00,
    #     patience=self.patience,
    #     mode="min",
    # )

    callbacks = [
        ckpt_callback,
        acc_pseudoLabels,
        prog_rate,
    ]
    
    return callbacks