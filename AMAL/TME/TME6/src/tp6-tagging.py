import itertools
import logging
from tqdm import tqdm
import torchmetrics as tm
import numpy as np
from datamaestro import prepare_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import List
import time
from icecream import ic

logging.basicConfig(level=logging.INFO)


class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement s'il n'est pas connu
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """

    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        """oov : autorise ou non les mots OOV"""
        self.oov = oov
        self.id2word = ["PAD"]
        self.word2id = {"PAD": Vocabulary.PAD}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]


class TaggingDataset:
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        self.sentences = []

        for s in data:
            self.sentences.append(
                (
                    [words.get(token["form"], adding) for token in s],
                    [tags.get(token["upostag"], adding) for token in s],
                )
            )

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, ix):
        return self.sentences[ix]


def collate_fn(batch):
    """Collate using pad_sequence"""
    return tuple(
        pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2)
    )


logging.info("Loading datasets...")
ds = prepare_dataset("org.universaldependencies.french.gsd")
# Format de sortie décrit dans
# https://pypi.org/project/conllu/
words = Vocabulary(True)
tags = Vocabulary(False)
train_data = TaggingDataset(ds.train, words, tags, True)
dev_data = TaggingDataset(ds.validation, words, tags, True)
test_data = TaggingDataset(ds.test, words, tags, False)


logging.info("Vocabulary size: %d", len(words))
logging.info("Tags size: %d", len(tags))
BATCH_SIZE = 100
BATCH_SIZE = 32
LEN_WORDS = len(words)
LEN_TAG = len(tags)
train_loader = DataLoader(
    train_data, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True
)
dev_loader = DataLoader(dev_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)


def run_epoch(
    loader, model, loss_fn, optimizer=None, logger=None, device="cuda", num_classes=18
):
    loss_list = []
    acc = tm.classification.Accuracy(task="multiclass", num_classes=num_classes)
    acc.to(device)
    model.to(device)
    model.train() if optimizer else model.eval()
    for input, target in loader:
        input = input.to(device)
        target = target.to(device)

        # ic(input.size())
        # ic(target.size())
        output = model(input)
        # ic(output.size())
        output = model.decode(output).transpose(1, 2)
        # ic(output.size())
        loss = loss_fn(output, target)
        loss_list.append(loss.item())
        acc(output.argmax(1), target)
        # backward if we are training
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return np.array(loss_list).mean(), acc.compute().item()


class Model(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_size,
        vocab_size,
        tag_size,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size)
        self.f_h = nn.Linear(hidden_size, tag_size)

    def forward(self, x):
        x = self.embedding(x)
        h, (_, _) = self.rnn(x)
        return h

    def decode(self, h):
        return self.f_h(h)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 0.001
nb_epoch = 10


model = Model(32, 64, LEN_WORDS, LEN_TAG)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in tqdm(range(nb_epoch)):
    mean_train_loss, acc_train = run_epoch(
        train_loader, model, loss_fn, optimizer, device=device
    )
    mean_test_loss, acc_test = run_epoch(test_loader, model, loss_fn, device=device)
    ic(mean_train_loss)
    ic(acc_train)
    ic(mean_test_loss)
    ic(acc_test)
