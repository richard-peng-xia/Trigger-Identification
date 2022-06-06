import re
import random
import json
import pandas as pd
from sklearn.utils import shuffle

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

import pytorch_lightning as pl
from nltk.tokenize import TweetTokenizer

tweet_tokenizer = TweetTokenizer()
# load tools for preprocessning text
with open("./tools/abbreviation.json", "r", encoding="utf-8") as f:
    abbre_dict = dict(json.load(f))

class TriggerDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self.data = data
        self.cids = pd.unique(data['cid']).tolist()
        self.cid_mid_dict = {cid: data[data['cid'] == cid].sort_values(by=['cid', 'time'])['mid'].tolist() for cid in self.cids}

    def __len__(self):
        return len(self.cids)

    def __getitem__(self, index):
        cid = self.cids[index]
        mids = self.cid_mid_dict.get(cid, [])
        data_cascade = self.data.loc[mids].to_dict(orient='records')
        return data_cascade

class TriggerDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.train_file = args.train_file
        self.test_file = args.test_file
        self.val_file = args.val_file
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.args = args

    def _load_file(self, filename, test=False):
        df = pd.read_csv(filename)
        if test:
            fields = ['mid', 'cid', 'pid', 'event', 'time', 'content']
            df = df[fields]
            df['trigger'] = [0]*len(df)
            df['verify'] = [0]*len(df)
        else:
            fields = ['mid', 'cid', 'pid', 'event', 'time', 'content', 'trigger', 'verify']
            df = df[fields]        
        # convert ID fields into string type to avoid incoherence in np/pd/torch
        df['cid'] = df['cid'].apply(str)
        df['mid'] = df['mid'].apply(str)
        df['pid'] = df['pid'].apply(str)
        # reset dataframe index
        df.index = df['mid']
        df.index.name = None
        return df

    def _load_dataset(self):
        self.df_train = self._load_file(self.train_file)
        self.df_val = self._load_file(self.val_file)
        self.df_test = self._load_file(self.test_file, test=True)

    def _clean_word(self, word):
        if word and word[0] != "@":
            # split with capitalized letter
            word = re.sub(r'([a-z]+|\d+)([A-Z])', r'\1 \2', word)
            word = word.lower()
        ## extend the abbreviated words
        word = " ".join([abbre_dict.get(sub_word, 0) if abbre_dict.get(sub_word, 0) else sub_word for sub_word in word.split(" ")])
        return word

    def _clean_sentence(self, sentence):
        '''function to clean single sentence'''
        sentence = re.sub('[hH]ttp\S+|www\.\S+', '', sentence)  # remove url        
        sentence = re.sub('<.*?>+', '', sentence) # remove html tags
        sentence = re.sub('@\S*', '<username>', sentence) # remove @
        # sentence = re.sub('#', '', sentence) # remove #
        # sentence = sentence.lower() # convert into lowercase
        ## preprocess:                         
        sentence = ' '.join(tweet_tokenizer.tokenize(sentence))      
        sentence = ' '.join([self._clean_word(word) for word in sentence.split()])   
        sentence = re.sub('\s[0-9]+\s', '', sentence) # remove numbers   
        sentence = re.sub('[\.\+\-\?\'\\,/$%&#:;^_`{|}~><“”]', '', sentence) # remove special tokens        
        # sentence = ' '.join([word_lemmatizer.lemmatize(word, pos='v') for word in sentence.split()])
        # sentence = ' '.join([word for word in sentence.split() if word not in stop_words])
        return sentence

    def _clean_text(self, text_field):
        print('Cleaning text...')
        clean_text_field = f'{text_field}_clean'
        self.df_train[clean_text_field] = self.df_train[text_field].apply(self._clean_sentence)
        self.df_val[clean_text_field] = self.df_val[text_field].apply(self._clean_sentence)
        self.df_test[clean_text_field] = self.df_test[text_field].apply(self._clean_sentence)

    def setup(self, stage=None):
        self._load_dataset()
        self._clean_text(text_field='content')  # clean text

    @staticmethod
    def collate_fn(item_list):
        flatten_item_list = [x for item in item_list for x in item]
        df_item = pd.DataFrame(flatten_item_list)
        batch = df_item.to_dict(orient='list')
        batch = {k: default_collate(v) for k, v in batch.items()}
        return batch

    def train_dataloader(self):
        return DataLoader(
            dataset=TriggerDataset(self.df_train),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=TriggerDataset(self.df_val),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=TriggerDataset(self.df_test),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

# data_module = TriggerDataModule(args)
# data_module.setup()
# loader_train = data_module.train_dataloader()
# data_module.df_train.iloc[0]