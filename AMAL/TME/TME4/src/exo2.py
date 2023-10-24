from icecream import ic, install

install()
from utils import RNN, device, SampleMetroDataset
import torch
import torchmetrics
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {device}")
# Nombre de stations utilisé
CLASSES = 2
# Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
# Taille du batch
BATCH_SIZE = 32
HIDDEN_SIZE = 20
PATH = "TME/TME4/data/"


matrix_train, matrix_test = torch.load(open(PATH + "hzdataset.pch", "rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = SampleMetroDataset(
    matrix_test[:, :, :CLASSES, :DIM_INPUT],
    length=LENGTH,
    stations_max=ds_train.stations_max,
)

# raise KeyError()
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

lr = 0.001
criterion = nn.CrossEntropyLoss()
total_epoch = 25
model = RNN(DIM_INPUT, HIDDEN_SIZE, CLASSES, batch_first=True)

#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence
model.to(device)
loss_train_per_epoch = []
loss_test_per_epoch = []
accuracy_train = torchmetrics.classification.Accuracy(
    task="multiclass", num_classes=CLASSES
)
accuracy_test = torchmetrics.classification.Accuracy(
    task="multiclass", num_classes=CLASSES
)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for epoch in tqdm(range(total_epoch)):
    epoch_loss = 0
    epoch_loss_test = 0
    for x, y in data_train:
        x = x.to(device)
        # ic(x.size())
        # ic(y.size())
        optimizer.zero_grad()

        h = torch.zeros(
            (x.size(0), HIDDEN_SIZE), device=device
        )  # (batch_size, hidden_size)
        h = model(x, h)
        y_hat = model.decode(h[:, -1])  # décone uniquement le dernier h
        # ic(y_hat.size())
        # ic(y.size())
        # ic(y_hat.size())
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.sum()
        accuracy_train(y_hat.argmax(1), y)

    # Eval
    with torch.no_grad():
        for x, y in data_test:
            x = x.to(device)
            h = torch.zeros(
                (x.size(0), HIDDEN_SIZE), device=device
            )  # (batch_size, hidden_size)
            h = model(x, h)
            y_hat = model.decode(h[:, -1])

            loss = criterion(y_hat, y)
            accuracy_test(y_hat.argmax(1), y)
            epoch_loss_test += loss.sum()
    loss_train_per_epoch.append(epoch_loss / len(data_train))
    loss_test_per_epoch.append(epoch_loss_test / len(data_test))
    print("step:", epoch)
    print("Loss_train:", float(loss_train_per_epoch[-1]))
    print("Loss_test:", float(loss_test_per_epoch[-1]))
    print("acc_train:", accuracy_train.compute())
    accuracy_train.reset()
    print("acc_test:", accuracy_test.compute())
    accuracy_test.reset()
