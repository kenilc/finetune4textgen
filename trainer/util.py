from functools import partial
import math
import random

import pandas as pd

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

class BucketByLengthSampler(torch.utils.data.Sampler):
    '''
    Equivalent to trax.data.inputs.BucketByLength.
    '''

    def __init__(self, dataset, boundaries, batch_sizes, length_fn):
        assert list(boundaries) == sorted(boundaries)
        assert len(boundaries) + 1 == len(batch_sizes)

        self.dataset = dataset
        self.boundaries = list(boundaries) + [float('inf')]
        self.batch_sizes = list(batch_sizes)
        self.length_fn = length_fn

    def __iter__(self):
        buckets = [[] for _ in range(len(self.batch_sizes))]
        for i, entry in enumerate(self.dataset):
            n = self.length_fn(entry)
            for j, b in enumerate(self.boundaries):
                if n <= b:
                    buckets[j].append(i)
                    break
        for bucket in buckets:
            random.shuffle(bucket)

        indices = list(range(len(buckets)))
        random.shuffle(indices)
        for i in indices:
            bucket = buckets[i]
            if bucket:
                batch_size = self.batch_sizes[i]
                n = math.ceil(len(bucket) / batch_size) * batch_size
                for j in range(0, n, batch_size):
                    yield bucket[j:j+batch_size]

    def __len__(self):
        return len(self.dataset)

def collate_fn(batch, pad_token_id):
    input_ids, labels = [], []
    for entry in batch:
        input_ids.append(torch.tensor(entry['input_ids']))
        labels.append(torch.tensor(entry['labels']))

    return {
        'input_ids': pad_sequence(input_ids, True, pad_token_id),
        'labels': pad_sequence(labels, True, pad_token_id)
    }

# Assume the source and target sequences are of similar lengths.
def length_fn(x):
    return len(x['input_ids']) + len(x['labels'])

def make_data_loader(dataset, tokenizer):
    # Tested on nvidia T4.
    boundaries  = [ 32, 128, 512, 2048]
    batch_sizes = [128,  32,   8,    2, 1]

    batch_sampler = BucketByLengthSampler(dataset, boundaries, batch_sizes, length_fn)

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
    )

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
            'labels': self.labels[idx]
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

def generate(tokenizer, model, prompt, **kwargs):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            **kwargs
        ).to('cpu')

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
