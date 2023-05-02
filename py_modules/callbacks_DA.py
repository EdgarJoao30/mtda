from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from copy import deepcopy
import matplotlib.pyplot as plt
import os
import torch
import numpy as np

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

    def on_train_epoch_end(self, trainer, pl_module):
        
        if pl_module.current_epoch == pl_module.max_epochs - 1:
            self.triggered_train.append(pl_module.current_epoch)
            # Create a subset of the dataframe with only the proportion columns
            prop_df = pl_module.pseudo_epoch_evolution[['epoch', 'prop_0', 'prop_1', 'prop_2', 'prop_3', 'prop_4', 'prop_5', 'prop_6', 'prop_7']]

            # Set up figure and axis
            fig, ax = plt.subplots(figsize=(15, 5))

            # Create stacked bar plot
            prop_df.plot(kind='bar', stacked=True, x='epoch', ax=ax)

            # Set axis labels and legend
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Proportion')
            ax.set_title('Proportion of Classes over Time')
            ax.legend(loc='upper left')
            
            self.pseudo_evolution_figure = fig
            self.pseudo_evolution_table = pl_module.pseudo_epoch_evolution
            
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
                monitor="train_combined_loss",
                dirpath=save_path,
                filename="ckpt-{epoch:02d}-{train_combined_loss:.2f}",
                save_top_k=1,
                mode="min",
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