import glob
import os
import os.path
import re
from tempfile import mkstemp
from urllib.parse import urlparse

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

def upload_to_gcs(folder, uri, override=False):
    o = urlparse(uri)
    if o.scheme != 'gs':
        raise ValueError('URI for Google Cloud Storage must starts with `gs://`')
    bucket_name, prefix = o.netloc, o.path[1:]

    regex = re.compile(fr'{folder}/*')
    bucket = storage.Client().get_bucket(bucket_name)
    for file_path in glob.glob(os.path.join(folder, '**'), recursive=True):
        if not os.path.isdir(file_path):
            truncated_path = regex.sub('', file_path)
            blob_name = f'{prefix}/{truncated_path}'
            blob = bucket.blob(blob_name)
            if override or not blob.exists():
                blob.upload_from_filename(file_path)

def generate(tokenizer, model, prompt, **kwargs):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            **kwargs
        ).to('cpu')

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
