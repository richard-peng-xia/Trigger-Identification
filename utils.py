import os
import sys

import pandas as pd

import torch

from torchmetrics.functional import accuracy, f1
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm, TQDMProgressBar
from pytorch_lightning.callbacks import BaseFinetuning

import wandb


class DisableValBar(TQDMProgressBar):
    '''customized callback to disable validation bar'''

    def __init__(self):
        super().__init__()

    def init_validation_tqdm(self):
        '''diable validation bar, disable = True'''
        has_main_bar = self.main_progress_bar is not None
        bar = Tqdm(
            desc="Validating",
            position=(2 * self.process_position + has_main_bar),
            disable=True,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar


class FineTune(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=0):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.model.encoder)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        # When `current_epoch` reaches the threshold, pre-trained LM will start training.
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.model.encoder,
                optimizer=optimizer,
                train_bn=True,
            )


class Evaluation:
    def __init__(self, args) -> None:
        self.name = args.name
        self.save_dir = args.save_dir
        self.time_now = args.time_now
        if 'record' not in os.listdir(args.save_dir):
            os.mkdir(os.path.join(args.save_dir, 'record'))

    def match(self, outputs, df, task=None):
        '''extract instance information'''
        task_info = f'{task}_' if task else ''
        # handling output in tensor-form
        preds = outputs[f'{task_info}preds']
        preds_class = torch.argmax(preds, dim=1)
        num_classes = preds.size(1)
        if task == 'trigger':
            ids = outputs['mid']
            ids_field = 'mid'
        elif task == 'verify':
            ids = outputs['cid']
            ids_field = 'cid'
        result = {f'{ids_field}': ids,'preds': preds_class.tolist()}
        return result