from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from copy import deepcopy
import os

class AccBestVal(LambdaCallback):
    def __init__(self):
        self.evolution = []

    def on_validation_epoch_end(self, trainer, pl_module):
        self.evolution.append(deepcopy(trainer.callback_metrics))

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

    acc_best_callback = AccBestVal()

    prog_rate = TQDMProgressBar(refresh_rate=10)

    # early_stop_callback = EarlyStopping(
    #     monitor="val_loss",
    #     min_delta=0.00,
    #     patience=self.patience,
    #     mode="min",
    # )

    callbacks = [
        ckpt_callback,
        acc_best_callback,
        prog_rate,
    ]
    
    return callbacks