```
usage: task.py [-h] [--model MODEL] [--epochs N] [--warmup-steps N] [--lr LR]
               [--log-interval N] [--snapshot-interval N] [--seed SEED]
               --train-data FILE [--test-data FILE] --log-dir DIR --model-dir DIR

T5 finetuner for seq2seq tasks

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         the pretrained T5 model to use (default: t5-small)
  --epochs N            number of epochs to train (default: 5)
  --warmup-steps N      number of warm-up steps (default: 200)
  --lr LR               learning rate (default: 0.005)
  --log-interval N      how many steps to wait before logging training status (default: 10)
  --snapshot-interval N
                        how many steps to wait before taking a model snapshot (default: 50)
  --seed SEED           random seed (default: 1)
  --train-data FILE     training dataset
  --test-data FILE      evaluation dataset
  --log-dir DIR         the directory to store the log
  --model-dir DIR       the directory to store the model
```
