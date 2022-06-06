from data import TriggerDataModule
from config import args
from model_wrapper import ModelWrapper
from utils import DisableValBar, FineTune, Evaluation

import wandb

import re
import os
import json
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything

from model_encoder.bert import Encoder
from model_interact.none import Interact
from model_integrate.mean import Integrate
from model_pipeline import Pipeline

if __name__ == '__main__':
    print(args.__dict__)

    data_module = TriggerDataModule(args)
    # data_module.setup()

    # set seed for model initialization and training
    seed_everything(args.ex_seed)
    # instantiate model wrapper
    model_wrapper = ModelWrapper(
        Encoder, Interact, Integrate, Pipeline, args)

    disable_val_bar = DisableValBar()
    fine_tune = FineTune(unfreeze_at_epoch=args.unfreeze_at_epoch)
    # initialize model checkpoint
    trigger_checkpoint = ModelCheckpoint(
        monitor='trigger_maF_val',
        dirpath=os.path.join(args.save_dir, 'checkpoint', f'{args.name}_trigger'),
        filename='trigger_{maF_val:.2f}_{epoch:02d}_{step:03d}',
        save_top_k=1,
        mode='max',
    )
    verify_checkpoint = ModelCheckpoint(
        monitor='verify_maF_val',
        dirpath=os.path.join(args.save_dir, 'checkpoint', f'{args.name}_verify'),
        filename='verify_{maF_val:.2f}_{epoch:02d}_{step:03d}',
        save_top_k=1,
        mode='max',
    )
    # initialize logger
    # wandb_logger = WandbLogger(
    #     save_dir=args.save_dir,       # directory to save log
    #     project=args.project,         # project name displayed in wandb
    #     name=args.name,               # run name displayed in wandb
    #     # version=args.model_des,     # model description
    #     log_model='False',            # log checkpoints all-during/True-at end/False-no
    #     config=args,                  # config to be reserved in wandb
    #     config_exclude_keys=[],       # string keys to exclude from config
    #     save_code=False,              # do not save code
    #     reinit=True,  # allow reinitializing runs to start multiple runs rom one script
    # )

    # tb_logger = TensorBoardLogger(
    #     save_dir= os.path.join(args.save_dir, 'tensorboard'),       # directory to save log
    #     name=args.project,
    #     version=args.name,
    # )

    # initialize trainer
    trainer = Trainer(
        callbacks=[trigger_checkpoint, verify_checkpoint,
                disable_val_bar, fine_tune],
        # logger=wandb_logger,
        # logger=tb_logger,
        default_root_dir=args.save_dir,
        # gpus=args.gpu,
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
    )

    # lr_finder = trainer.tuner.lr_find(
    #     model_wrapper,
    #     datamodule=data_module,
    # )
    # new_lr = lr_finder.suggestion()  # pick suggested learning rate
    # print(f'suggested lr: {new_lr}')
    # model_wrapper.lr = new_lr  # update hparams of the model
    # args.lr = new_lr

    # training
    # trainer.fit(model_wrapper, data_module)

    # fetch best model
    # def fetch_best(checkpoint_callback=None, checkpoint_path=None, task=None):
    #     if checkpoint_callback:
    #         best_path = checkpoint_callback.best_model_path
    #         best_score = checkpoint_callback.best_model_score.item()
    #     elif checkpoint_path:
    #         best_path = checkpoint_path
    #         best_score = eval(re.findall(r"[\d\.]+", best_path)[0])
    #     best_epoch = int(re.findall(r"epoch=\d+", best_path)[0].split("=")[1])
    #     best_step = int(re.findall(r"step=\d+", best_path)[0].split("=")[1])
    #     fetch_info = {
    #         f'{task}_best_path': best_path,
    #         f'{task}_best_epoch': best_epoch,
    #         f'{task}_best_step': best_step,
    #         f'{task}_best_score': best_score,
    #     }
    #     return fetch_info

    # trigger_fetch_info = fetch_best(checkpoint_callback=trigger_checkpoint, task='trigger')
    # verify_fetch_info = fetch_best(checkpoint_callback=verify_checkpoint, task='verify')

    # model_trigger = model_wrapper.load_from_checkpoint(
    #     checkpoint_path=trigger_fetch_info['trigger_best_path'])
    # model_verify = model_wrapper.load_from_checkpoint(
    #     checkpoint_path=verify_fetch_info['verify_best_path'])

    model_trigger = model_wrapper.load_from_checkpoint(
        checkpoint_path="./save/checkpoint/20220331-195531_bigbert_exseed=1_trigger/trigger_maF_val=0.00_epoch=09_step=3398.ckpt")
    model_verify = model_wrapper.load_from_checkpoint(
        checkpoint_path="./save/checkpoint/20220331-195531_bigbert_exseed=1_verify/verify_maF_val=0.00_epoch=11_step=4112.ckpt")


    # testing
    trainer.test(model_trigger, data_module)
    trigger_outputs = model_trigger.test_outputs
    trainer.test(model_verify, data_module)
    verify_outputs = model_verify.test_outputs

    df_test = data_module.test_dataloader().dataset.data

    eval_tool = Evaluation(args)
    instance_trigger = eval_tool.match(trigger_outputs, df_test, task='trigger')
    instance_verify = eval_tool.match(verify_outputs, df_test, task='verify')

    instance_all = {'trigger':instance_trigger, 'verify':instance_verify}
    record_file = 'test_result.json'
    with open(record_file, 'w') as f:
        json.dump(instance_all, f)
        print(f'Test result has been saved at: {record_file}')