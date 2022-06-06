from datasets import cached_path
import torch
import torch.nn as nn
from torch.nn import functional as F
import transformers
from transformers import BertTokenizer, BertModel, BigBirdTokenizer, BigBirdModel
transformers.logging.set_verbosity_error()


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pretrain_file = args.pretrain_file   
        self.tokenizer = BigBirdTokenizer.from_pretrained(self.pretrain_file, cache_dir="./pretrained_model")
        self.bert = BigBirdModel.from_pretrained(self.pretrain_file, return_dict=True, cache_dir="./pretrained_model")
        # self.tokenizer = tokenizer_temp
        # self.bert = bert_temp
        # temp parameters to obtain model device name
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.sent_size = args.bert_embedding_size

    def forward(self, mes):
        '''Encode textual mesence into embeddings
        mes(list of str, e.g. ['xxx']): strings of messages
        '''
        # extract mesence representation with pre-trained LM
        inputs = self.tokenizer(
            mes, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.dummy_param.device) for k, v in inputs.items()}
        bert_outputs = self.bert(**inputs)
        mes_embed = bert_outputs.last_hidden_state[:, 0, :]
        return mes_embed

# encoder = Encoder(args)
# fields = ['mid', 'cid', 'pid', 'time', 'content_clean', 'trigger', 'verify']
# mids, cids, pids, time, mes, yt, yv = tuple([batch_data1[field] for field in fields])
# structure = (mids, cids, pids, time)
# mes_embed = encoder(mes)
