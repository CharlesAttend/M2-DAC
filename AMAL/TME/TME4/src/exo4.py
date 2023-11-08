import string
import unicodedata
import torch
import sys
from tqdm import tqdm
from icecream import ic, install

install()
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from utils import RNN, device

## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation + string.digits + " "
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1, len(LETTRES) + 1), LETTRES))
id2lettre[0] = ""  ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(), id2lettre.keys()))
PATH = "TME/TME4/data/"


def normalize(s):
    """Nettoyage d'une chaîne de caractères."""
    return "".join(c for c in unicodedata.normalize("NFD", s) if c in LETTRES)


def string2code(s):
    """Transformation d'une chaîne de caractère en tenseur d'indexes"""
    return torch.tensor([lettre2id[c] for c in normalize(s)])


def code2string(t):
    """Transformation d'une liste d'indexes en chaîne de caractères"""
    if type(t) != list:
        t = t.tolist()
    return "".join(id2lettre[i] for i in t)


class TrumpDataset(Dataset):
    def __init__(self, text, maxsent=None, maxlen=None):
        """Dataset pour les tweets de Trump
        * text : texte brut
        * maxsent : nombre maximum de phrases.
        * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [
            p[:maxlen].strip() + "." for p in full_text.split(".") if len(p) > 0
        ]
        if maxsent is not None:
            self.phrases = self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, i):
        # print(self.phrases[i])
        t = string2code(self.phrases[i])
        # print(t)
        t = torch.cat([torch.zeros(self.MAX_LEN - t.size(0), dtype=torch.long), t])
        return t[:-1], t[1:]
        # Ok en faite ça donne un vecteur de taille maxlen (paddé par des zéro) où chaque caractère de la phrase est un nombre


# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 1
DIM_OUTPUT = len(id2lettre)
BATCH_SIZE = 64
HIDDEN_SIZE = 20
lr = 0.005
total_epoch = 33
MAX_LEN = 50

criterion = nn.CrossEntropyLoss()
data_trump = DataLoader(
    TrumpDataset(
        open(PATH + "trump_full_speech.txt", "rb").read().decode(), maxlen=MAX_LEN
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)  # Pas de train / test ?
softmax = nn.Softmax(2)
embeding = nn.Linear(DIM_OUTPUT, DIM_OUTPUT).to(device)
model = RNN(
    DIM_OUTPUT, HIDDEN_SIZE, DIM_OUTPUT, batch_first=True
)  # Avec entrée = onehot
model.to(device)
loss_train_per_epoch = []
loss_test_per_epoch = []
params = list(model.parameters()) + list(embeding.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
def train(model_name):
    for epoch in tqdm(range(total_epoch)):
        epoch_loss = 0
        epoch_loss_test = 0
        for x, y in tqdm(data_trump):
            # (32, 808) => (32, 808, len(lettre2id)), float pour que ça passe dans les Linears
            y = nn.functional.one_hot(y, num_classes=DIM_OUTPUT).to(device).float()
            x = nn.functional.one_hot(x, num_classes=DIM_OUTPUT).to(device).float()
            x = x.to(device).float()
            y = y.to(device).float()
            x = embeding(x) # 1 epoch 4 min à la place de 85s avec ça horrible 
            # x = x.to(device).float().unsqueeze(2)
            # y = y.to(device).float()
            # x = x.unsqueeze
            # x = x.unsqueeze(2)
            # ic(x.size())
            # ic(y.size())
            optimizer.zero_grad()

            h = torch.zeros(
                (x.size(0), HIDDEN_SIZE), device=device
            )  # (batch_size, hidden_size)
            h = model(x, h)
            y_hat = model.decode(h)
            # y_hat = softmax(y_hat).argmax(2)  # softmax sur dim 2
            # y_hat = y_hat.argmax(2).float() 
            y_hat = y_hat.float()
            # ic(y_hat)
            # ic(y)
            # ic(y_hat.size())
            # ic(y.size())
            loss = criterion(y_hat, y)
            # ic(loss)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.sum()

        loss_train_per_epoch.append(epoch_loss / len(data_trump))
        print("step:", epoch)
        print("Loss_train:", float(loss_train_per_epoch[-1]))
        print(code2string(y[0,:,:].argmax(1)))
        print(code2string(y_hat[0,:,:].argmax(1)))
    torch.save(model.state_dict(), f"TME/TME4/{model_name}.pt")
    torch.save(embeding.state_dict(), f"TME/TME4/{model_name}_embeding.pt")


def generate(model_name, lenght=45, prompt='Trump:'):
    model.load_state_dict(torch.load(f"TME/TME4/{model_name}.pt"))
    embeding.load_state_dict(torch.load(f"TME/TME4/{model_name}_embeding.pt"))
    for _ in range(50):
        h = torch.zeros(
            (1, HIDDEN_SIZE), device=device
        )
        generated = [torch.tensor(torch.randint(len(lettre2id), (1,))).to(device)]
        for i in range(MAX_LEN):
            h = model.one_step(embeding(nn.functional.one_hot(generated[-1], num_classes=len(lettre2id)).float()), h)
            generated.append(model.decode(h).argmax(1))
        generated = torch.stack(generated[1:])
        print("".join([id2lettre[int(i)] for i in generated.squeeze()]))
    """
    x = string2code(prompt)
    x = torch.cat([torch.zeros(MAX_LEN - x.size(0), dtype=torch.long), x])
    x = nn.functional.one_hot(x, num_classes=DIM_OUTPUT).to(device).float()
    x = x.unsqueeze(0) # Batch de 1
    ic(x.size())
    model = RNN(
        DIM_OUTPUT, HIDDEN_SIZE, DIM_OUTPUT, batch_first=True
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    output = []
    for i in range(lenght):
        h = torch.zeros(
            (x.size(0), HIDDEN_SIZE), device=device
        )  # (batch_size, hidden_size)
        h = model(x, h)
        y_hat = model.decode(h)
        y_hat = softmax(y_hat).argmax(2).squeeze()
        output += y_hat.tolist()
    print(len(output))
    print(code2string(output))
    """
    
model_name = "model_oh"
torch.save(model.state_dict(), f"TME/TME4/{model_name}.pt")
train(model_name)
generate(model_name)
    
    
    
