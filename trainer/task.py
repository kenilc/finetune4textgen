from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    HfArgumentParser
)
from transformers.data.data_collator import DataCollatorForSeq2Seq
from datasets import Dataset

def load_dataset(data_source, tokenizer):
    def _tokenize(entry):
        output = {
            'input_ids': tokenizer.encode(entry['source']),
            'labels': tokenizer.encode(entry['target'])
        }
        output['length'] = max(len(output['input_ids']), len(output['labels']))
        return output

    dataset = Dataset.from_csv(data_source)
    dataset = dataset.map(_tokenize)
    return dataset

def run(training_args, remaining_args):
    model = T5ForConditionalGeneration.from_pretrained(remaining_args.model_name)
    tokenizer = T5Tokenizer.from_pretrained(remaining_args.model_name)

    train_dataset = load_dataset(remaining_args.train_data, tokenizer)

    eval_dataset = load_dataset(remaining_args.eval_data, tokenizer) \
        if remaining_args.eval_data else None

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
        '--train_data',
        default=None,
        required=True,
        help='training dataset (CSV format)'
    )
    parser.add_argument(
        '--eval_data',
        default=None,
        help='evaluation dataset (CSV format)'
    )
    # Ignore unknown args.
    return parser.parse_args_into_dataclasses(return_remaining_strings=True)[:2]

if __name__ == '__main__':
    training_args, remaining_args = get_args()
    run(training_args, remaining_args)
