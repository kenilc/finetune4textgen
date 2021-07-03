import pandas as pd

import torch
from torch.utils.data import Dataset

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    HfArgumentParser
)
from transformers.data.data_collator import DataCollatorForSeq2Seq

class Seq2SeqDataset(Dataset):
    def __init__(self, data_source, tokenizer):
        data = self.load_data(data_source)
        self.input_ids = tokenizer(data['source'].tolist())['input_ids']
        self.labels = tokenizer(data['target'].tolist())['input_ids']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.labels[idx],
        }

    def load_data(self, data_source):
        data = None
        if type(data_source) == str:
            if data_source.endswith('csv'):
                data = pd.read_csv(data_source)
            if data_source.endswith('pickle'):
                data = pd.read_pickle(data_source)
            if data_source.endswith('parquet'):
                data = pd.read_parquet(data_source)
        elif type(data_source) == pd.core.frame.DataFrame:
            data = data_source.copy()
        if data is None:
            raise ValueError('Data source must be a pandas.DataFrame or a file (CSV, pickle, parquet)')
        if 'source' not in data.columns or 'target' not in data.columns:
            raise ValueError('Data must have columns `source` and `target`')
        return data

def run(training_args, remaining_args):
    model = T5ForConditionalGeneration.from_pretrained(remaining_args.model_name)
    tokenizer = T5Tokenizer.from_pretrained(remaining_args.model_name)

    train_dataset = Seq2SeqDataset(remaining_args.train_data, tokenizer)

    test_dataset = Seq2SeqDataset(remaining_args.eval_data, tokenizer) \
        if remaining_args.eval_data else None

    data_collator = DataCollatorForSeq2Seq(tokenizer, model, padding='longest')

    trainer = Trainer(
        model         = model,
        tokenizer     = tokenizer,
        args          = training_args,
        data_collator = data_collator,
        train_dataset = train_dataset,
        eval_dataset  = test_dataset,
    )
    trainer.train()

def get_args():
    parser = HfArgumentParser(TrainingArguments)
    parser.add_argument(
        '--model_name',
        default='t5-small',
        help='pre-trained model name (default: t5-small)'
    )
    parser.add_argument(
        '--train_data',
        default=None,
        required=True,
        help='training dataset'
    )
    parser.add_argument(
        '--eval_data',
        default=None,
        help='evaluation dataset'
    )
    return parser.parse_args_into_dataclasses(return_remaining_strings=True)

if __name__ == '__main__':
    training_args, remaining_args = get_args()[:2]
    run(training_args, remaining_args)
