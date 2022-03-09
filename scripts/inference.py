import pickle
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional import (
    accuracy,
    f1_score,
    precision_recall
)
from tqdm import tqdm
from transformers import BertTokenizer

from src.bert_classifier import BertClassifier
from src.data import (
    collate_fn,
    TextDataset,
)


def predict(
    model: BertClassifier,
    loader: DataLoader,
) -> Tuple[np.ndarray, float, float, float, float]:
    preds = []
    model.eval()
    acc = 0
    f1 = 0
    prec = 0
    rec = 0
    n = 0

    for batch in tqdm(loader):
        for key in batch:
            batch[key] = batch[key].to(model.device)

        labels = batch.pop('Class')

        with torch.no_grad():
            pred = model(batch).logits.argmax(axis=1)

        if labels.size()[1] > 0:
            acc += accuracy(
                pred,
                labels,
            )
            f1 += f1_score(
                pred,
                labels,
            )
            _prec, _rec = precision_recall(
                pred,
                labels,
            )
            prec += _prec
            rec += _rec

        preds.append(pred.cpu().numpy())
        n += len(pred)

    return (
        np.concatenate(preds),
        acc / n,
        f1 / n,
        prec / n,
        rec / n,
    )


if __name__ == '__main__':
    parser = ArgumentParser('infer model on validation set')
    parser.add_argument(
        '-f',
        '--file',
        type=Path,
        required=True,
        help='path to validation .csv file',
    )
    parser.add_argument(
        '-c',
        '--ckpt',
        type=Path,
        required=True,
        help='path to model checkpoint',
    )
    parser.add_argument(
        '-l',
        '--le',
        type=Path,
        required=True,
        help='path to pickled LabelEncoder',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='batch size',
    )

    args = parser.parse_args()
    file = args.file
    ckpt = args.ckpt
    le = args.le
    batch_size = args.batch_size

    if not file.exists() or not file.is_file():
        raise ValueError('validation file does not exist or is not a file')

    if not ckpt.exists() or not ckpt.is_file():
        raise ValueError('checkpoint does not exist or is not a file')

    if not le.exists() or not le.is_file():
        raise ValueError('pickled LabelEncoder does not exist or is not a file')

    with open(le, 'rb') as f:
        label_encoder = pickle.load(f)

    tokenizer = BertTokenizer.from_pretrained('sberbank-ai/ruBert-base')
    collate_with_tokenizer = partial(
        collate_fn,
        tokenizer=tokenizer,
    )

    df = pd.read_csv(file).head(64)
    dataset = TextDataset(
        df,
        label_encoder,
    )
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_with_tokenizer,
    )

    model = BertClassifier.load_from_checkpoint(
        ckpt,
        map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        model_name_or_path='sberbank-ai/ruBert-base',
    )

    predictions, acc, f1, prec, rec = predict(
        model,
        test_loader,
    )

    print(
        f'accuracy:  {acc:.2f}\n'
        f'f1:        {f1:.2f}\n'
        f'precision: {prec:.2f}\n'
        f'recall:    {rec:.2f}\n',
    )

    submission = pd.DataFrame()
    submission['Id'] = df['Id']
    submission['Class'] = label_encoder.inverse_transform(predictions)

    submission.to_csv(
        'submission.csv',
        index=False,
    )
