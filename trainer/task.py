import argparse
from datetime import datetime
import os
import os.path

import torch
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AdamW,
    get_scheduler
)

from .util import (
    Seq2SeqDataset,
    make_data_loader
)

import logging
logging.basicConfig(
    format='%(asctime)-15s %(levelname)s %(message)s',
    level=logging.INFO
)

def train(args):
    torch.manual_seed(args.seed)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logging.info(f'device: {device}')

    model = T5ForConditionalGeneration.from_pretrained(args.model)
    model = model.to(device).train()
    logging.info(f"Pre-trained model '{args.model}' loaded")

    tokenizer = T5Tokenizer.from_pretrained(args.model)
    logging.info(f'Tokenizer loaded')

    train_dataset = Seq2SeqDataset(args.train_data, tokenizer)
    train_data_loader = make_data_loader(train_dataset, tokenizer)
    logging.info(f'Training data loaded')

    test_data_loader = None
    if args.test_data:
        test_dataset = Seq2SeqDataset(args.test_data, tokenizer)
        test_data_loader = make_data_loader(test_dataset, tokenizer)
        logging.info(f'Test data loaded')

    optimizer = AdamW(model.parameters(), lr=args.lr)

    lr_scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.epochs * len(train_data_loader)
    )

    logging.info(f'Training started')
    with SummaryWriter(args.log_dir) as writer:
        steps = 0
        for epoch in range(args.epochs):
            for batch in train_data_loader:
                optimizer.zero_grad()
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                if steps % args.log_interval == 0:
                    writer.add_scalar('Loss/train', loss, steps)

                    if test_data_loader:
                        test_loss = test(model, device, test_data_loader)
                        writer.add_scalar('Loss/test', test_loss, steps)
                        logging.info(f'steps: {steps} loss/train: {loss} loss/test: {test_loss}')

                if steps > 0 and steps % args.snapshot_interval == 0:
                    save(model, 'SNAPSHOT', args.model_dir)

                optimizer.step()
                lr_scheduler.step()
                steps += 1
    logging.info(f'Training ended')

    save(model, 'FINAL', args.model_dir)

def test(model, device, test_data_loader):
    loss, num_batches = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in test_data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss += outputs.loss
            num_batches += 1
    model.train()
    return loss / num_batches

def save(model, tag, model_dir):
    ts = datetime.now().isoformat()
    path = os.path.join(model_dir, f'model-{tag}-{ts}.pt')
    torch.save(model, path)

def get_args():
    parser = argparse.ArgumentParser(description='T5 finetuner for seq2seq tasks')
    parser.add_argument(
        '--model',
        default='t5-small',
        help='the pretrained T5 model to use (default: t5-small)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        metavar='N',
        help='number of epochs to train (default: 5)'
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=200,
        metavar='N',
        help='number of warm-up steps (default: 200)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.005,
        metavar='LR',
        help='learning rate (default: 0.005)'
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='how many steps to wait before logging training status (default: 10)'
    )
    parser.add_argument(
        '--snapshot-interval',
        type=int,
        default=50,
        metavar='N',
        help='how many steps to wait before taking a model snapshot (default: 50)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='random seed (default: 1)'
    )
    parser.add_argument(
        '--train-data',
        default=None,
        metavar='FILE',
        required=True,
        help='training dataset'
    )
    parser.add_argument(
        '--test-data',
        default=None,
        metavar='FILE',
        help='evaluation dataset'
    )
    parser.add_argument(
        '--log-dir',
        default=None,
        metavar='DIR',
        required=True,
        help='the directory to store the log'
    )
    parser.add_argument(
        '--model-dir',
        default=None,
        metavar='DIR',
        required=True,
        help='the directory to store the model'
    )

    args, _ = parser.parse_known_args()
    return args

def init(args):
    assert os.path.exists(args.train_data)
    assert args.test_data is None or os.path.exists(args.test_data)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

if __name__ == '__main__':
    args = get_args()
    init(args)
    train(args)
