import torch
from pytorch_lightning.callbacks import Callback

import os


class PretrainCheckpointCallback(Callback):
    def on_save_checkpoint(self, trainer, pl_module):
        checkpoint_filename = (
            "-".join(
                [
                    "pretrain",
                    pl_module.hparams.arch,
                    pl_module.hparams.dataset,
                    pl_module.hparams.comment,
                ]
            )
            + ".cp"
        )
        checkpoint_path = os.path.join(pl_module.hparams.checkpoint_dir, checkpoint_filename)
        torch.save(pl_module.model.state_dict(), checkpoint_path)
