from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    HfArgumentParser
)
from transformers.data.data_collator import DataCollatorForSeq2Seq
from datasets import Dataset

def load_dataset(path_or_paths, tokenizer):
    def _tokenize(entry):
        output = {
            'input_ids': tokenizer.encode(entry['source']),
            'labels': tokenizer.encode(entry['target'])
        }
        output['length'] = max(len(output['input_ids']), len(output['labels']))
        return output

    path_or_paths = [path.strip() for path in path_or_paths.split(',')]

    dataset = Dataset.from_csv(path_or_paths)
    dataset = dataset.map(_tokenize)
    return dataset

def run(training_args, custom_args):
    model = T5ForConditionalGeneration.from_pretrained(custom_args.model_name)
    tokenizer = T5Tokenizer.from_pretrained(custom_args.model_name)

    train_dataset = load_dataset(custom_args.train_csv_files, tokenizer)

    eval_dataset = load_dataset(custom_args.eval_csv_files, tokenizer) \
        if custom_args.eval_csv_files else None

    data_collator = DataCollatorForSeq2Seq(tokenizer, model, padding='longest')

    trainer = Trainer(
        model         = model,
        tokenizer     = tokenizer,
        args          = training_args,
        data_collator = data_collator,
        train_dataset = train_dataset,
        eval_dataset  = eval_dataset,
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
        '--train_csv_files',
        default=None,
        required=True,
        help='CSV file(s) for training. Paths are comma-separated'
    )
    parser.add_argument(
        '--eval_csv_files',
        default=None,
        help='CSV file(s) for evalation. Paths are comma-separated'
    )
    # Ignore unknown args.
    return parser.parse_args_into_dataclasses(return_remaining_strings=True)[:2]

if __name__ == '__main__':
    training_args, custom_args = get_args()
    run(training_args, custom_args)
