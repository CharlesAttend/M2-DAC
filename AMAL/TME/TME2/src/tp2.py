import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
from tqdm import tqdm


writer = SummaryWriter()

data = datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()
datax = torch.tensor(datax, dtype=torch.float)
datay = torch.tensor(datay, dtype=torch.float).reshape(-1, 1)
input_size = 13
hidden_size = 20
output_size = 1
# TODO:


# Question 2
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.linear1(x)
        out = self.tanh(out)
        out = self.linear2(out)
        return out


model = nn.Sequential(
    nn.Linear(input_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, output_size)
)

# Définir la fonction de perte (MSE)
criterion = nn.MSELoss()

# Définir l'optimiseur (par exemple, Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000
for epoch in range(num_epochs):
    outputs = model(datax)
    loss = criterion(outputs, datay)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)
    writer.add_scalar("Loss/train", loss / len(datax), epoch)
