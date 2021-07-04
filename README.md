A small python script to fine-tune a pre-trained T5 model for seq-2-seq task.

(You may easily modify `trainer/task.py` for other models.

## Sample usage

Please refer to 
[TrainingArguments](https://huggingface.co/transformers/_modules/transformers/training_args.html) 
for the complete list of available arguments and their meaning.

### How to fine-tune

You may also check out this repo and run it on Google colab notebooks using the `%%bash` magic.

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
    --group_by_length \
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
* `transformers`
* `sentencepiece`
* `pytorch`
