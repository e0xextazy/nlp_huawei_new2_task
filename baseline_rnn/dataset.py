import re
from collections import Counter
from itertools import chain
from typing import List

import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset, DataLoader


class Tokenizer:
    def __init__(self, word_pattern="[\w']+"):
        """
        Simple tokenizer that splits the sentence by given regex pattern
        :param word_pattern: pattern that determines word boundaries
        """
        self.word_pattern = re.compile(word_pattern)

    def tokenize(self, text):
        return self.word_pattern.findall(text)


class Vocab:
    def __init__(self, tokenized_texts: List[List[str]], max_vocab_size=None):
        """
        Builds a vocabulary by concatenating all tokenized texts and counting words.
        Most common words are placed in vocabulary, others are replaced with [UNK] token
        :param tokenized_texts: texts to build a vocab
        :param max_vocab_size: amount of words in vocabulary
        """
        counts = Counter(chain(*tokenized_texts))
        max_vocab_size = max_vocab_size or len(counts)
        common_pairs = counts.most_common(max_vocab_size)
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.EOS_IDX = 2
        self.itos = ["<PAD>", "<UNK>", "<EOS>"] + [pair[0]
                                                   for pair in common_pairs]
        self.stoi = {token: i for i, token in enumerate(self.itos)}

    def vectorize(self, text: List[str]):
        """
        Maps each token to it's index in the vocabulary
        :param text: sequence of tokens
        :return: vectorized sequence
        """
        return [self.stoi.get(tok, self.UNK_IDX) for tok in text]

    def __iter__(self):
        return iter(self.itos)

    def __len__(self):
        return len(self.itos)


class TextDataset(Dataset):
    def __init__(self, tokenized_texts, labels, vocab: Vocab):
        """
        A Dataset for the task
        :param tokenized_texts: texts from a train/val/test split
        :param labels: corresponding toxicity ratings
        :param vocab: vocabulary with indexed tokens
        """
        self.texts = tokenized_texts
        self.labels = labels
        self.vocab = vocab

    def __getitem__(self, item):
        return (
            self.vocab.vectorize(self.texts[item]) + [self.vocab.EOS_IDX],
            self.labels[item],
        )

    def __len__(self):
        return len(self.texts)

    def collate_fn(self, batch):
        """
        Technical method to form a batch to feed into recurrent network
        """
        tmp = pack_sequence(
            [torch.tensor(pair[0]) for pair in batch], enforce_sorted=False
        ), torch.tensor([pair[1] for pair in batch])
        return tmp


def train_test_split(data, train_frac=0.85):
    """
    Splits the data into train and test parts, stratifying by labels.
    Should it shuffle the data before split?
    :param data: dataset to split
    :param train_frac: proportion of train examples
    :return: texts and labels for each split
    """
    n_toxicity_ratings = 5
    train_labels = []
    val_labels = []
    train_texts = []
    val_texts = []
    for label in range(n_toxicity_ratings):
        texts = data[data.rate == label].text.values
        n_train = int(len(texts) * train_frac)
        n_val = len(texts) - n_train
        train_texts.extend(texts[:n_train])
        val_texts.extend(texts[n_train:])
        train_labels += [label] * n_train
        val_labels += [label] * n_val
    return train_texts, train_labels, val_texts, val_labels
