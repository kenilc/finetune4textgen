import os
from tempfile import mkstemp

from datasets import Dataset
from google.cloud import storage

import torch

def load_dataset(path_or_paths, tokenizer):
    def _tokenize(entry):
        output = {
            'input_ids': tokenizer.encode(entry['source']),
            'labels': tokenizer.encode(entry['target'])
        }
        output['length'] = max(len(output['input_ids']), len(output['labels']))
        return output

    local_files = []
    temp_files = []
    for path in path_or_paths.split(','):
        path = path.strip()
        if path.startswith('gs://'):
            temp_file = download_from_gcs(path)
            temp_files.append(temp_file)
        else:
            local_files.append(path)

    dataset = Dataset.from_csv(local_files + temp_files)
    dataset = dataset.map(_tokenize)

    for temp_file in temp_files:
        os.unlink(temp_file)

    return dataset

def download_from_gcs(path):
    fd, temp_file = mkstemp()
    with os.fdopen(fd, 'wb') as f:
        storage.Client().download_blob_to_file(path, f)
    return temp_file

def generate(tokenizer, model, prompt, **kwargs):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            **kwargs
        ).to('cpu')

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
