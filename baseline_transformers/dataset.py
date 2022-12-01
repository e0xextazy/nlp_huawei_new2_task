import pandas as pd
import torch
from torch.utils.data import Dataset


class RottenTomatoesDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_seq_len):
        self.data = dataframe
        self.text = dataframe['movie_description']
        self.targets = None
        if 'target' in dataframe:
            self.targets = dataframe['target']
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __getitem__(self, index):
        text = str(self.text[index])
        text = ' '.join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        if self.targets is not None:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'targets': torch.tensor(self.targets[index], dtype=torch.long)
            }
        else:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
            }

    def __len__(self) -> int:
        return len(self.text)


def train_test_split(data: pd.DataFrame, train_frac=0.85):
    """
    Splits the data into train and test parts, stratifying by labels.
    Should it shuffle the data before split?
    :param data: dataset to split
    :param train_frac: proportion of train examples
    :return: texts and labels for each split
    """
    # n_films_genres = 6
    train_data, test_data = None, None
    train_texts = []
    test_texts = []
    train_labels = []
    test_labels = []

    for label in data['target'].unique():
        texts = data[data.target == label].movie_description
        n_train = int(len(texts) * train_frac)
        n_test = len(texts) - n_train
        train_texts.extend(texts[:n_train])
        test_texts.extend(texts[n_train:])
        train_labels += [label] * n_train
        test_labels += [label] * n_test
        train_data = {
            'movie_description': train_texts,
            'target': train_labels
        }
        test_data = {
            'movie_description': test_texts,
            'target': test_labels
        }
    return pd.DataFrame(train_data), pd.DataFrame(test_data)
