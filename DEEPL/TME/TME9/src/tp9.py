import logging
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
from icecream import ic
from datamaestro import prepare_dataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


class FolderText(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""

    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix

        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text() if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        s = self.files[ix]
        return (
            self.tokenizer(s if isinstance(s, str) else s.read_text()),
            self.filelabels[ix],
        )


def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText)

    """
    WORDS = re.compile(r"\S+")

    words, embeddings = prepare_dataset(
        "edu.stanford.glove.6b.%d" % embedding_size
    ).load()
    OOVID = len(words)
    words.append("__OOV__")

    word2id = {word: ix for ix, word in enumerate(words)}
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")
    ds = prepare_dataset("edu.stanford.aclimdb")

    return (
        word2id,
        embeddings,
        FolderText(ds.train.classes, ds.train.path, tokenizer, load=False),
        FolderText(ds.test.classes, ds.test.path, tokenizer, load=False),
    )
word2id, embeddings, train_dataset, test_dataset = get_imdb_data()
embeddings = np.array(embeddings)
#  TODO:

import torch
from torch import nn
from torch.utils.data import DataLoader

from ignite.engine import (
    Engine,
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine, wandb_logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Input layer to hidden layer
            nn.ReLU(),  # Activation function (ReLU)
            nn.Linear(hidden_size, hidden_size),  # Hidden layer to output layer
            nn.ReLU(),  # Activation function (ReLU)
            nn.Linear(hidden_size, hidden_size),  # Hidden layer to output layer
            nn.ReLU(),  # Activation function (ReLU)
            nn.Linear(hidden_size, output_size),  # Hidden layer to output layer
        )

    def forward(self, x):
        x = x.mean(0).view(-1, 1)
        x = self.model(x)
        return x


# Example usage:
input_size = 50  # Input size for MNIST dataset (28x28 images)
hidden_size = 128  # Number of neurons in the hidden layer
output_size = 2  # Number of output classes for classification
epochs = 10
lr = 0.005

model = MLP(input_size, hidden_size, output_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

def collate_fn(batch):
    """Collate using pad_sequence"""
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        embedded_text = torch.tensor([embeddings[i] for i in _text])
        text_list.append(embedded_text)
    return pad_sequence(text_list), torch.tensor(label_list)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)


trainer = create_supervised_trainer(model, optimizer, criterion, device)
val_metrics = {
    "accuracy": Accuracy(),
    "loss": Loss(criterion)
}

train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    train_evaluator.run(train_loader)
    metrics = train_evaluator.state.metrics
    print(f"Training Results - Epoch {trainer.state.epoch} - Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    val_evaluator.run(val_loader)
    metrics = val_evaluator.state.metrics
    print(f"Validation Results - Epoch[{trainer.state.epoch}] - Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")

# Define a Tensorboard logger
tb_logger = TensorboardLogger(log_dir="tb-logger")

# Attach handler to plot trainer's loss every 100 iterations
tb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED(every=100),
    tag="training",
    output_transform=lambda loss: {"batch_loss": loss},
)

# Attach handler for plotting both evaluators' metrics after every epoch completes
for tag, evaluator in [("training", train_evaluator), ("validation", val_evaluator)]:
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag=tag,
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer),
    )
trainer.run(train_loader, max_epochs=epochs)
