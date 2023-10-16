from pathlib import Path
import os
import torch
from tqdm import tqdm

from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime

# Téléchargement des données

from datamaestro import prepare_dataset

ds = prepare_dataset("com.lecun.mnist")
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels = ds.test.images.data(), ds.test.labels.data()

# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs/runs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyDataset(Dataset):
    def __init__(self, data, label) -> None:
        super().__init__()
        self.data = data.reshape((len(data), len(data[0]) ** 2)) / 255
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


class AutoEncoderUSPS(nn.Module):
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def __init__(self, n_hidden, dim_x=28 * 28):
        super().__init__()

        self.fc1 = nn.Linear(dim_x, n_hidden, dtype=torch.float64)
        self.fc2 = nn.Linear(n_hidden, dim_x, dtype=torch.float64)
        self.fc2.weight.requires_grad = False

        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def encoder(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x

    def decoder(self, x):
        # Same Weigth (sinon ça prends plus de temps à converger, c'est une meilleurs initialisation, plus proche de la réalité)
        x = F.linear(x, self.fc1.weight.t(), self.fc2.bias)
        x = self.sig(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class State:
    def __init__(self, model, optimizer) -> None:
        self.model = model
        self.optimizer = optimizer
        self.epoch = 0
        self.iteration = 0


""" 
    ┌────────────────────────────────────────────────────────────────────────┐
    │ Parameter init                                                         │
    └────────────────────────────────────────────────────────────────────────┘
"""
device = "cpu"
print(f"running on {device}")
savepath = Path("model.pch")
lr = 0.01
criterion = nn.MSELoss()
n_hidden = 20
batch_size = 32
total_epoch = 15

train_dataset = MyDataset(train_images, train_labels)
test_dataset = MyDataset(test_images, test_labels)
train_loader = DataLoader(
    train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True
)
sample_loader = DataLoader(train_dataset, batch_size=8, pin_memory=True)
test_loader = DataLoader(
    test_dataset, shuffle=True, batch_size=batch_size, pin_memory=True
)



""" 
    ┌────────────────────────────────────────────────────────────────────────┐
    │ Training loop with state management                                    │
    └────────────────────────────────────────────────────────────────────────┘
"""
# State management
# if savepath.is_file():
#     with savepath.open("rb") as fp:
#         state = torch.load(fp)
# else:
#     model = AutoEncoderUSPS(n_hidden=n_hidden)
#     model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     state = State(model, optimizer)

# for epoch in range(state.epoch, total_epoch):
#     epoch_loss = 0
#     epoch_loss_test = 0
#     for index in range(state.iteration, len(train_loader)):
#         x, labels = train_loader[index]
#         state.optim.zero_grad()
#         x = x.to(device)

#         outputs = state.model(x)
#         # loss = criterion(outputs, labels)
#         loss = criterion(outputs, x) # x because autoencodeur
#         epoch_loss += loss.sum()

#         loss.backward()
#         state.optimizer.step()
#         state.iteration += 1
        
#         # Eval
#         with torch.no_grad():
#             for x, _ in test_loader:
#                 x = x.to(device)
#                 outputs = model(x)
#                 loss = criterion(outputs, x)
#                 epoch_loss_test += loss.sum()
#         writer.add_scalar("Loss/train", epoch_loss / len(train_loader), epoch)
#         writer.add_scalar("Loss/test", epoch_loss_test / len(test_loader), epoch)
        
#     with savepath.open("wb") as fp:
#         state.epoch += 1
#         torch.save(state, fp)
    

""" 
    ┌────────────────────────────────────────────────────────────────────────┐
    │ Clasic training loop                                                   │
    └────────────────────────────────────────────────────────────────────────┘
"""
model = AutoEncoderUSPS(n_hidden=n_hidden)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
state = State(model, optimizer)
for epoch in tqdm(range(total_epoch)):
    epoch_loss = 0
    epoch_loss_test = 0
    for x, _ in train_loader:
        x = x.to(device)
        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, x)
        epoch_loss += loss.sum()

        loss.backward()
        optimizer.step()

    # Eval
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            outputs = model(x)
            loss = criterion(outputs, x)
            epoch_loss_test += loss.sum()
    writer.add_scalar("Loss/train", epoch_loss / len(train_loader), epoch)
    writer.add_scalar("Loss/test", epoch_loss_test / len(test_loader), epoch)

for x, _ in sample_loader:
    images = x.reshape(8, 28, 28).unsqueeze(1).repeat(1, 3, 1, 1).double()
    images = make_grid(images)
    writer.add_image(f"samples/original", images, 0)
    x = x.to(device)
    outputs = model(x)
    break
images = outputs.reshape((8, 28, 28)).unsqueeze(1).repeat(1, 3, 1, 1)
images = make_grid(images)
writer.add_image(f"samples/Reconstruction last epoch", images, 0)
