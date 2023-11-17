# %%
import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch.nn as nn
import torchmetrics as tm
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
from icecream import ic
import unicodedata
import string
from tqdm import tqdm
from pathlib import Path
from typing import List


import wandb
import time
import re
from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(level=logging.INFO)

FILE = "../data/en-fra.txt"

writer = SummaryWriter("/tmp/runs/tag-" + time.asctime())


def normalize(s):
    return re.sub(
        " +",
        " ",
        "".join(
            c if c in string.ascii_letters else " "
            for c in unicodedata.normalize("NFD", s.lower().strip())
            if c in string.ascii_letters + " " + string.punctuation
        ),
    ).strip()


class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """

    PAD = 0
    EOS = 1
    SOS = 2
    OOVID = 3

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {
            "PAD": Vocabulary.PAD,
            "EOS": Vocabulary.EOS,
            "SOS": Vocabulary.SOS,
        }
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


class TradDataset:
    def __init__(self, data, vocOrig, vocDest, adding=True, max_len=10):
        self.sentences = []
        for s in tqdm(data.split("\n")):
            if len(s) < 1:
                continue
            orig, dest = map(normalize, s.split("\t")[:2])
            if len(orig) > max_len:
                continue
            self.sentences.append(
                (
                    torch.tensor(
                        [vocOrig.get(o) for o in orig.split(" ")] + [Vocabulary.EOS]
                    ),
                    torch.tensor(
                        [vocDest.get(o) for o in dest.split(" ")] + [Vocabulary.EOS]
                    ),
                )
            )

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        return self.sentences[i]


def collate_fn(batch):
    orig, dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig), o_len, pad_sequence(dest), d_len


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(FILE) as f:
    lines = f.readlines()

lines = [lines[x] for x in torch.randperm(len(lines))]
idxTrain = int(0.8 * len(lines))

vocEng = Vocabulary(True)
vocFra = Vocabulary(True)
MAX_LEN = 100
BATCH_SIZE = 32

datatrain = TradDataset("".join(lines[:idxTrain]), vocEng, vocFra, max_len=MAX_LEN)
datatest = TradDataset("".join(lines[idxTrain:]), vocEng, vocFra, max_len=MAX_LEN)

train_loader = DataLoader(
    datatrain, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    datatest, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False
)

#  TODO:  Implémenter l'encodeur, le décodeur et la boucle d'apprentissage

# %%
class Encoder(nn.Module):
    def __init__(self, input_vocab_size, hidden_size, embedding_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size)

    def forward(self, input, input_lengths):
        embedded = self.embedding(input)
        # packed = pack_padded_sequence(embedded, input_lengths, enforce_sorted=False)
        # output, h_n = self.gru(packed)
        return self.gru(embedded)


class Decoder(nn.Module):
    def __init__(
        self, output_vocab_size, hidden_size, embedding_dim, max_length=MAX_LEN
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.embedding = nn.Embedding(output_vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.to_vocab = nn.Linear(hidden_size, output_vocab_size)

    def one_step(self, input, hidden):
        """
        Input est soit
        * Mode contraint : Les vrais mots de la phrase
        * Mode non contraint : Le mot précédent prédit par le décodeur
        """
        output = self.embedding(input)
        # ic(output.size())
        # output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.to_vocab(output)
        return output, hidden

    def forward(self, encoder_outputs, encoder_hidden, lens_seq, target_tensor=None):
        batch_size = encoder_outputs.size(1)
        decoder_input = torch.empty(
            1, batch_size, dtype=torch.long, device=device
        ).fill_(Vocabulary.SOS)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(lens_seq):
            # ic()
            # ic(decoder_input.size())
            # ic(decoder_hidden.size())
            decoder_output, decoder_hidden = self.one_step(
                decoder_input, decoder_hidden
            )

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[i, :].unsqueeze(0)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(
                    -1
                ).detach()  # detach from history as input
            decoder_outputs.append(decoder_output)

        decoder_outputs = torch.cat(decoder_outputs, dim=0)
        # decoder_outputs = nn.functional.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden


def run_epoch(
    loader,
    encoder,
    decoder,
    loss_fn,
    num_classes,
    optimizer=None,
    scheduler=None,
    logger=None,
    device="cuda",
):
    loss_list = []
    acc = tm.classification.Accuracy(
        task="multiclass", num_classes=num_classes, ignore_index=Vocabulary.PAD
    )
    acc.to(device)
    encoder.to(device)
    decoder.to(device)

    if optimizer:
        encoder.train()
        decoder.train()
    else:
        encoder.eval()
        decoder.eval()

    for x, len_x, y, len_y in loader:
        coin_flip = int(torch.rand(1))  # stay on teacher forcing mode yet
        x = x.to(device)
        y = y.to(device)

        # Encoder part
        encoder_outputs, encoder_hidden = encoder(x, len_x)
        # Decoder part
        if coin_flip:  # teacher forcing mode
            decoder_outputs, _ = decoder(
                encoder_outputs, encoder_hidden, y.size(0), target_tensor=y
            )
        else:
            decoder_outputs, _ = decoder(encoder_outputs, encoder_hidden, y.size(0))

        # y_oh = nn.functional.one_hot(y, num_classes=num_classes).float()
        # Pour éviter la transformation en OneHot [18, 32, 22904] => [18, 22904, 32]
        # Comme ça ça fit le y [18, 32]
        decoder_outputs = decoder_outputs.transpose(1, 2)
        # ic(y.size())
        # ic(decoder_outputs.size())
        loss = loss_fn(decoder_outputs, y)
        loss_list.append(loss.item())
        acc(decoder_outputs.argmax(1), y)

        # backward if we are training
        if optimizer:
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
            loss.backward()
            optimizer[0].step()
            optimizer[1].step()
            
            if scheduler:
                scheduler[0].step()
                scheduler[1].step()
    pred_sentence = vocFra.getwords(decoder_outputs.argmax(1)[:, 0])
    print(
        f"Original sentence {vocEng.getwords(x[:, 0])}\n",
        f"Predicted sentence {pred_sentence}\n",
        f"True Sentence {vocFra.getwords(y[:,0])}\n",
    )
    return np.array(loss_list).mean(), acc.compute().item()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 0.0025
lr_encoder = lr
lr_decoder = lr
nb_epoch = 50

hidden_size = 64
embded_size = 64

scheduler = optim.lr_scheduler.ExponentialLR
gamma = 0.95

wandb.init(
    # set the wandb project where this run will be logged
    project="amal",
    # track hyperparameters and run metadata
    config={
        "TME": "TME6",
        "learning_rate": lr,
        "hidden_size": hidden_size,
        "embded_size": embded_size,
        "epochs": nb_epoch,
        "scheduler": scheduler.__name__
        "scheduler_gamma": gamma
    },
)


len_voc_origin = len(vocEng)
len_voc_dest = len(vocFra)
loss_fn = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD)

encoder = Encoder(len_voc_origin, hidden_size, embded_size)
decoder = Decoder(len_voc_dest, hidden_size, embded_size)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr_encoder)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr_decoder)
scheduler_encoder = scheduler(encoder_optimizer, gamma=0.95)
scheduler_decoder = scheduler(decoder_optimizer, gamma=0.95)
# optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
# optimizer.add_param_group(decoder.parameters())
for epoch in tqdm(range(nb_epoch)):
    mean_train_loss, acc_train = run_epoch(
        train_loader,
        encoder,
        decoder,
        loss_fn,
        len_voc_dest,
        optimizer=(encoder_optimizer, decoder_optimizer),
        scheduler=(scheduler_encoder, scheduler_decoder),
        device=device,
    )
    mean_test_loss, acc_test = run_epoch(
        test_loader, encoder, decoder, loss_fn, len_voc_dest, device=device
    )
    torch.save(encoder, f"encoder_{hidden_size}_{embded_size}.pt")
    torch.save(decoder, f"decoder_{hidden_size}_{embded_size}.pt")
    wandb.log(
        {
            "mean_train_loss": mean_train_loss,
            "acc_train": acc_train,
            "mean_test_loss": mean_test_loss,
            "acc_test": acc_test,
        }
    )
    ic(mean_train_loss)
    ic(acc_train)
    ic(mean_test_loss)
    ic(acc_test)
wandb.finish()

# %%



