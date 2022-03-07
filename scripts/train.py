import warnings
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.bert_classifier import BertClassifier
from src.data import (
    collate_fn,
    TextDataset,
)

CWD = Path(__file__).parents[1]
DEFAULT_LOG_DIR = CWD / 'pl-logs'

warnings.simplefilter(
    'ignore',
    UserWarning,
)


if __name__ == '__main__':
    parser = ArgumentParser('train model for binary classification')
    parser.add_argument(
        '-d',
        '--data',
        required=True,
        type=Path,
        help='directory with data',
    )
    parser.add_argument(
        '-o',
        '--out',
        required=True,
        type=Path,
        help='directory for resulting model',
    )
    parser.add_argument(
        '-l',
        '--logs',
        default=DEFAULT_LOG_DIR,
        help='directory for PL logs',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='batch size',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='number of workers for DataLoaders',
    )

    args = parser.parse_args()
    data_dir: Path = args.data
    out_dir = args.out
    logs_dir = args.logs

    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError('data directory does not exist or is not a directory')

    out_dir.mkdir(
        exist_ok=True,
        parents=True,
    )
    logs_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    train = pd.read_csv(data_dir / 'train.csv')
    test = pd.read_csv(data_dir / 'test.csv')
    val = pd.read_csv(data_dir / 'val.csv')

    le = LabelEncoder().fit(train['Class'].values)
    train = TextDataset(
        train,
        le=le,
    )
    val = TextDataset(
        val,
        le=le,
    )
    test = TextDataset(
        test,
        le=le,
    )

    batch_size = args.batch_size
    num_epochs = args.epochs
    num_workers = args.num_workers

    tokenizer = BertTokenizer.from_pretrained('sberbank-ai/ruBert-large')
    model = BertClassifier(
        'sberbank-ai/ruBert-large',
        num_labels=len(le.classes_),
    )

    collate_with_tokenizer = partial(
        collate_fn,
        tokenizer=tokenizer,
    )
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_with_tokenizer,
    )
    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_with_tokenizer,
    )
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_with_tokenizer,
    )

    logger = pl.loggers.TensorBoardLogger(
        save_dir=out_dir,
        name='lightning_logs',
    )
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=num_epochs,
        num_sanity_val_steps=1
    )
    trainer.fit(
        model,
        train_loader,
        val_loader,
    )
    trainer.save_checkpoint(out_dir / 'model.ckpt')
