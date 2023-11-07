import itertools
import logging
from tqdm import tqdm
import torchmetrics as tm
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

ds = prepare_dataset("org.universaldependencies.french.gsd")


# Format de sortie décrit dans
# https://pypi.org/project/conllu/


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


def train_epoch(loader, model, loss_fn, optimizer=None, logger=None, cuda=False):
    model.train()
    loss_list = []
    acc = tm.classification.Accuracy(task="multiclass", num_classes=5)
    for input, target in loader:
        if cuda:  # only with GPU, and not with CPU
            input = input.cuda()
            target = target.cuda()

        ic(input.size())
        ic(target.size())
        # forward
        output = model(input)
        loss = criterion(output, target)
        loss_list.append(loss.item)
        # backward if we are training
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return np.array(loss_list).mean()


def evaluate(loader, model, loss_fn, cuda=False):
    model.eval()
    acc = tm.classification.Accuracy(task="multiclass", num_classes=5)
    for input, target in loader:
        if cuda:  # only with GPU, and not with CPU
            input = input.cuda()
            target = target.cuda()

        # forward
        output = model(input)
        loss = criterion(output, target)


class Model(nn.Module):
    def __init__(
        self,
        hidden_size,
    ):
        super().__init__()
        self.rnn = nn.LSTM(10, 20, 2)

    def forward(self, x):
        return self.rnn(x)


logging.info("Loading datasets...")
words = Vocabulary(True)
tags = Vocabulary(False)
train_data = TaggingDataset(ds.train, words, tags, True)
dev_data = TaggingDataset(ds.validation, words, tags, True)
test_data = TaggingDataset(ds.test, words, tags, False)


logging.info("Vocabulary size: %d", len(words))


cuda = torch.cuda.is_available()
BATCH_SIZE = 100

lr = 0.001
nb_epoch = 10


train_loader = DataLoader(
    train_data, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True
)
dev_loader = DataLoader(dev_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)
model = Model(10)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in tqdm(range(nb_epoch)):
    mean_loss, acc_test = train_epoch(train_loader, model, loss_fn, cuda=cuda)
    acc_test = evaluate(test_loader, model, loss_fn, cuda=cuda)
