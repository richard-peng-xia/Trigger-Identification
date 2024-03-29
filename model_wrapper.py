﻿import torch

import pytorch_lightning as pl
from transformers import AdamW

from torchmetrics.functional import accuracy, f1


class ModelWrapper(pl.LightningModule):
    def __init__(self, Encoder, Interact, Integrate, Pipeline, args):
        super().__init__()
        # inherited args
        self.args = args
        self.lr = args.lr

        # model layers
        self.model = Pipeline(Encoder, Interact, Integrate, args)

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    def forward(self, batch_data):
        '''input batch data to produce output'''
        output = self.model(batch_data)
        return output

    def configure_optimizers(self):
        r'''set model optimizer'''
        # filter the parameters based on `requires_grad` to ensure parameter-freeze callback
        optimizer = AdamW(filter(lambda p: p.requires_grad,
                          self.model.parameters()), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        r'''operation in training step'''
        output = self(batch)
        # compute loss and metrics
        loss = output['trigger_loss'] + output['verify_loss']
        log_dict = self._handle_output(output, mode='train')
        # print(log_dict) # debug using [print] because [self.log_dict] is callable only when trainer is initialized with logger
        self.log_dict(log_dict)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        '''operation in validation step'''
        output = self(batch)
        return output

    def validation_epoch_end(self, outputs_list):
        outputs = self._stack_outputs(outputs_list)
        # compute loss and metrics
        log_dict = self._handle_output(outputs, mode='val')
        self.log_dict(log_dict)

    def test_step(self, batch, batch_idx):
        '''operation in validation step'''
        output = self(batch)
        return output

    def test_epoch_end(self, outputs_list) -> None:
        outputs = self._stack_outputs(outputs_list)
        self.test_outputs = outputs

    def _stack_outputs(self, outputs_list):
        '''stack output lists'''
        outputs = {}
        for index, output in enumerate(outputs_list):
            for k, v in output.items():
                record_k = outputs.get(k, None)
                if k == 'loss':
                    continue
                if record_k is None:
                    outputs[k] = v
                elif isinstance(v, torch.Tensor):
                    if len(v.shape) == 0:
                        outputs[k] = (record_k*index+v)/(index+1)
                    else:
                        outputs[k] = torch.cat([record_k, v], dim=0)
                else:
                    outputs[k] = record_k + v
        return outputs

    def _handle_output(self, output, mode='train'):
        '''handle loss and metrics   mode: train/val/test'''
        trigger_loss = output['trigger_loss'].item()
        trigger_preds = output['trigger_preds']
        trigger_targets = output['trigger_targets']
        trigger_acc = accuracy(
            trigger_preds, trigger_targets)
        trigger_maF = f1(
            trigger_preds, trigger_targets,
            average='macro', num_classes=trigger_preds.size(1))
        verify_loss = output['verify_loss'].item()
        verify_preds = output['verify_preds']
        verify_targets = output['verify_targets']
        verify_acc = accuracy(
            verify_preds, verify_targets)
        verify_maF = f1(
            verify_preds, verify_targets,
            average='macro', num_classes=verify_preds.size(1))
        log_dict = {
            f'trigger_loss_{mode}': trigger_loss,
            f'trigger_acc_{mode}': trigger_acc,
            f'trigger_maF_{mode}': trigger_maF,
            f'verify_loss_{mode}': verify_loss,
            f'verify_acc_{mode}': verify_acc,
            f'verify_maF_{mode}': verify_maF,
        }
        return log_dict
