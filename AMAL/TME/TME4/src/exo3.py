from utils import RNN, device, ForecastMetroDataset
from icecream import ic, install
install()
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch, torchmetrics

# Nombre de stations utilisé
CLASSES = 3
# Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
# Taille du batch
BATCH_SIZE = 32
HIDDEN_SIZE = 20
PATH = "TME/TME4/data/"


matrix_train, matrix_test = torch.load(open(PATH + "hzdataset.pch", "rb"))
ds_train = ForecastMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = ForecastMetroDataset(
    matrix_test[:, :, :CLASSES, :DIM_INPUT],
    length=LENGTH,
    stations_max=ds_train.stations_max,
)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

lr = 0.01
criterion = torch.nn.MSELoss()
total_epoch = 33 # Avant overfit
model = RNN(DIM_INPUT, HIDDEN_SIZE, DIM_INPUT, batch_first=True)

#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence
model.to(device)
loss_train_per_epoch = []
loss_test_per_epoch = []
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for epoch in tqdm(range(total_epoch)):
    epoch_loss = 0
    epoch_loss_test = 0
    for x, y in data_train:
        x.to(device)
        for i in range(CLASSES):
            x_station_i, y_station_i = x[:, :, i, :], y[:, :, i, :]
            x_station_i = x_station_i.to(device)
            # ic(x_station_i.size())
            # ic(y.size())
            optimizer.zero_grad()

            h = torch.zeros(
                (x_station_i.size(0), HIDDEN_SIZE), device=device
            )  # (batch_size, hidden_size)
            h = model(x_station_i, h)
            y_hat = model.decode(h)  # décone uniquement le dernier h
            # ic(y_station_i.size())
            # ic(y_hat.size())
            loss = criterion(y_hat, y_station_i)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.sum()

    # Eval
    with torch.no_grad():
        for x, y in data_test:
            x.to(device)
            for i in range(CLASSES):
                x_station_i, y_station_i = x[:, :, i, :], y[:, :, i, :]
                x_station_i = x_station_i.to(device)

                h = torch.zeros(
                    (x_station_i.size(0), HIDDEN_SIZE), device=device
                )  # (batch_size, hidden_size)
                h = model(x_station_i, h)
                y_hat = model.decode(h)  # décone uniquement le dernier h
                loss = criterion(y_hat, y_station_i)
                epoch_loss_test += loss.sum()

    loss_train_per_epoch.append(epoch_loss / len(data_train))
    loss_test_per_epoch.append(epoch_loss_test / len(data_test))
    print("step:", epoch)
    print("Loss_train:", float(loss_train_per_epoch[-1]))
    print("Loss_test:", float(loss_test_per_epoch[-1]))
