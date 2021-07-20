A small python script to fine-tune pre-trained models for text generation. The script heavily uses the built-in functionalities of [Hugging Face's transformers](https://huggingface.co).

## Sample usage

The script can be run locally or on Google AI Platform.
Similarly, training and evaluation data can be stored locally or downloaded from Google Cloud Storage.

Please refer to
[TrainingArguments](https://huggingface.co/transformers/_modules/transformers/training_args.html)
for the complete list of available arguments and their meaning.

### How to fine-tune

On local machines:

```
python trainer/task.py \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --learning_rate 1e-5 \
    --warmup_steps 500 \
    --num_train_epochs 5 \
    --log_level info \
    --logging_dir "$LOG_DIR" \
    --logging_strategy steps \
    --logging_steps 250 \
    --save_strategy steps \
    --save_steps 5000 \
    --group_by_length True \
    --output_dir "$OUTPUT_DIR" \
    --train_csv_files "$DATA_DIR/train_data.csv" \
    --eval_csv_files "$DATA_DIR/eval_data.csv" \
    --model_name "t5-small"
```

The last 3 arguments are additional for this script.

### Data format for training and evaluation

CSV files must have two columns "source" and "target". For example,
to fine-tune a pre-trained T5 model that will make a sentence from
a given list of words, the data may look like:

```
source,target
"make sentence with: Tom, eat, apple.","Tom ate an apple."
"make sentence with: Ada, dad, go, book, store, book.","Ada and her dad went to the book store to buy a book."
```

### How to use the fine-tuned model

```
from transformers import T5ForConditionalGeneration T5Tokenizer
from trainer.util import generate

checkpoint = '/output/checkpoint-XXXXX/'

model = T5ForConditionalGeneration.from_pretrained(checkpoint)
tokenizer = T5Tokenizer.from_pretrained(checkpoint)

prompt = 'make sentence with: John, car.'
generate(tokenizer, model, prompt)
```

will (hopefully 😅) output

```
John was driving a car.
```

## Required Python packages

* `datasets`
* `google-cloud-storage`
* `sentencepiece`
* `torch`
* `transformers`

## References

* [Exploring Transfer Learning with T5: the Text-To-Text Transfer Transformer](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)
