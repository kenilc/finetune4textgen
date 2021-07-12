from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    HfArgumentParser
)
from transformers.data.data_collator import DataCollatorForSeq2Seq

from .util import load_dataset, UploadToGCSCallback

def run(training_args, custom_args):
    model = AutoModelForSeq2SeqLM.from_pretrained(custom_args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(custom_args.model_name)

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
        callbacks     = [
            UploadToGCSCallback(custom_args.job_dir)
        ]
    )
    trainer.train()

def get_args():
    parser = HfArgumentParser(TrainingArguments)
    parser.add_argument(
        '--model_name',
        default='t5-small',
        help=(
            'Pre-trained model name (default: t5-small), or'
            ' path to a saved checkpoint to resume training'
        )
    )
    parser.add_argument(
        '--train_csv_files',
        default=None,
        required=True,
        help=(
            'CSV file(s) for training. Paths are comma-separated.'
            ' Files can be local or in Google Cloud Storage'
        )
    )
    parser.add_argument(
        '--eval_csv_files',
        default=None,
        help=(
            'CSV file(s) for evalation. Paths are comma-separated.'
            ' Files can be local or in Google Cloud Storage'
        )
    )
    parser.add_argument(
        '--job-dir',
        default=None,
        help='For training on Google AI platform'
    )
    # Ignore unknown args.
    return parser.parse_args_into_dataclasses(return_remaining_strings=True)[:2]

if __name__ == '__main__':
    training_args, custom_args = get_args()
    run(training_args, custom_args)
