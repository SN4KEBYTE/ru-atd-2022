from typing import (
    Optional,
    Tuple,
    Union,
)

import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    BatchEncoding,
    BertTokenizer,
)


def collate_fn(
    input_data,
    tokenizer: BertTokenizer,
) -> BatchEncoding:
    texts, labels = zip(*input_data)
    labels = torch.LongTensor(labels)

    inputs = tokenizer(
        texts,
        return_tensors='pt',
        padding='longest',
        max_length=256,
        truncation=True,
    )
    inputs['Class'] = labels

    return inputs


class TextDataset(Dataset):
    def __init__(
        self,
        data,
        le: Optional[LabelEncoder] = None,
    ) -> None:
        self.texts = data['Text'].values

        if 'Class' in data.columns:
            self.labels = data['Class'].values if le is None else le.transform(data['Class'])

    def __len__(
        self,
    ) -> int:
        return len(self.texts)

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[str, Union[str, list]]:
        if hasattr(self, 'labels'):
            return self.texts[idx], self.labels[idx]
        else:
            return self.texts[idx], []
