from datetime import datetime, timedelta
import os

class Args:
    def __init__(self):
        # environment
        self.train_file = './data_trigger/train.csv'
        self.val_file = './data_trigger/val.csv'
        self.test_file = './data_trigger/test2.csv'
        self.pretrain_file = 'google/bigbird-roberta-base' # local file of pretrained model or use online version 'bert-base-uncased'
        self.project = 'trigger_identification'     # project name for wandb logging
        self.model_name = 'bigbert'            # model description
        self.save_dir = './save'  # save directory for logging information
        self.num_workers = 8      # num of cpus of the remote machine used in dataloader
        self.gpu = ['cpu']            # the gpu indice to train models
        # model parameters (general)
        self.sent_size = 768      # sentence size
        self.hidden_size = 300    # hidden unit size
        # model parameters (model-specific)
        ## encoder: bert
        self.bert_embedding_size = 768
        # hyper-parameters
        self.dropout = 0.1    # dropout rate
        self.ex_seed = 1      # experiment seed
        self.lr = 2e-5        # initial learning rate (pytorch_lightning can automatically choose suitable lr)
        self.batch_size = 4   # the num of cascades in a batch
        self.max_epochs = 25  # training epochs
        self.unfreeze_at_epoch = 0  # num of epochs for LM fine-tuning, default as 0 meaning always update LM parameters
        # experiment
        self.log_every_n_steps = 4
        self.val_check_interval = 4


args = Args()

# initialize directory to save checkpoint
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
# create starting time
args.time_now = (datetime.now()+timedelta(hours=8)).strftime('%Y%m%d-%H%M%S') # chinese time
# create run name
ex_info = f'exseed={args.ex_seed}'
args.name = f"{args.time_now}_{args.model_name}_{ex_info}"
print(f"run name: {args.name}")
