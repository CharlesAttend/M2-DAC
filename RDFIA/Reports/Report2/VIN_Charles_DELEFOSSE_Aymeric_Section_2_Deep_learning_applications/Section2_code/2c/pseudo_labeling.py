# %%
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision
import wandb
from sklearn.manifold import TSNE
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

print("imported")

# %% [markdown]
# We are going to do a **domain adaptation** from a source dataset (MNIST) towards a target dataset (MNIST-M).
#
# First, we need to create the target dataset:

from mnistm import create_mnistm

create_mnistm()

# %% [markdown]
# Then, let's load the MNIST dataset and compute its (train!) mean and standard deviation.
#
# We will use those values to **standardize** both MNIST and MNIST-M.

# %%
mnist_pixels = torchvision.datasets.MNIST(".", train=True, download=True).data / 255
mean = mnist_pixels.mean().item()
std = mnist_pixels.std().item()

print(f"Mean {mean} and Std {std}")
mean = torch.tensor([mean, mean, mean])
std = torch.tensor([std, std, std])

# %% [markdown]
# Create the loaders for MNIST...

# %%
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x,
        transforms.Normalize(mean, std),
    ]
)

mnist_train = torchvision.datasets.MNIST(".", train=True, transform=transform)
mnist_test = torchvision.datasets.MNIST(".", train=False, transform=transform)

source_train_loader = DataLoader(mnist_train, batch_size=128)
source_test_loader = DataLoader(mnist_test, batch_size=128)

# %%

# %%
with open("mnistm_data.pkl", "rb") as f:
    mnist_m = pickle.load(f)


class MNISTM(torch.utils.data.Dataset):
    def __init__(self, x, y, transform):
        self.x, self.y = x, y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        x = self.transform(x)

        return x, y


mnistm_train = MNISTM(mnist_m["x_train"], mnist_m["y_train"], transform)
mnistm_test = MNISTM(mnist_m["x_test"], mnist_m["y_test"], transform)

target_train_loader = DataLoader(mnistm_train, batch_size=128)
target_test_loader = DataLoader(mnistm_test, batch_size=128)


# %%
class NaiveNet(nn.Module):
    def __init__(self):
        super().__init__()  # Important, otherwise will throw an error

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),  # (32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # (32, 14, 14)
            nn.Conv2d(32, 48, kernel_size=5, stride=1, padding=2),  # (48, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  #  (48, 7, 7)
        )

        self.classif = nn.Sequential(
            nn.Linear(48 * 7 * 7, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            # Softmax include in the Cross-Entropy loss
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.flatten(start_dim=1)
        x = self.classif(x)
        return x


NaiveNet()(torch.randn(2, 3, 28, 28)).shape


# %%
def eval_model(net, loader):
    net.eval()

    acc, loss = 0, 0.0
    c = 0
    for x, y in loader:
        c += len(x)

        with torch.no_grad():
            logits = net(x.cuda()).cpu()

        loss += F.cross_entropy(logits, y).item()
        acc += (logits.argmax(dim=1) == y).sum().item()

    return round(100 * acc / c, 2), round(loss / len(loader), 5)


# %% [markdown]
# Let's train our naive model, but only the source (MNIST) dataset. We will evaluate its performance on the target (MNIST-M) dataset afterwards.
#
# Notice that we use a **learning rate scheduler**. We are updating the learning rate after each epoch according to a function defined with a *lambda* following the paper specification.
#
# We set the initial learning rate to 1.0 because `LambdaLR` defines a *multiplicative factor* of the base learning rate.
#
# It's often useful to reduce likewise the learning rate during training, to facilitate convergence once the model has found a good local minima (we rarely find the global).

# %%
epochs = 10

naive_net = NaiveNet().cuda()

optimizer = torch.optim.SGD(naive_net.parameters(), lr=1.0, momentum=0.9)

mu0, alpha, beta = 0.01, 10, 0.75
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda e: 0.01 / (1 + alpha * e / epochs) ** beta
)

total_batches = epochs * len(source_train_loader)
with tqdm(total=total_batches, desc=f"Epoch 1/{epochs}", unit="batch") as pbar:
    for epoch in range(epochs):
        train_loss = 0.0

        for x, y in source_train_loader:
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            logits = naive_net(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pbar.update(1)

        pbar.set_description(f"Epoch {epoch + 1}/{epochs}")
        scheduler.step()
        pbar.set_postfix(
            {
                "train loss": round(train_loss / len(source_train_loader), 5),
                "learning rate": optimizer.param_groups[0]["lr"],
            }
        )
        # print(f"\tLearning rate = {optimizer.param_groups[0]['lr']}")

    test_acc, test_loss = eval_model(naive_net, source_test_loader)
    print(f"Test loss: {test_loss}, test acc: {test_acc}")

# %% [markdown]
# Performance on less than 10 epochs are great on MNIST, more than 99% accuracy! But this dataset is quite easy.
#
# Now, the real question is: can our model generalize on the slightly different domain of MNIST-M?

# %%
test_acc, test_loss = eval_model(naive_net, target_test_loader)
print(f"Test loss: {test_loss}, test acc: {test_acc}")

# %%
source_train_loader = DataLoader(mnist_train, batch_size=64)
target_train_loader = DataLoader(mnistm_train, batch_size=64)


# %%
def extract_emb(net, loader):
    embeddings = []

    for x, _ in loader:
        with torch.no_grad():
            feats = net.cnn(x.cuda()).view(len(x), -1).cpu()

        embeddings.append(feats.numpy())

    return np.concatenate(embeddings)


# %%
def plot_t_sne(emb_2d, domains, title=None, save_name=None):
    # plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=domains)
    df = pd.DataFrame(
        {
            "x": emb_2d[:, 0],
            "y": emb_2d[:, 1],
            "Domains": np.where(
                domains, "Source embeddings", "Target domain embeddings"
            ),
        }
    )
    sns.scatterplot(
        data=df, x="x", y="y", hue="Domains", palette="deep", linewidth=0, s=10
    )
    if title:
        plt.gca().set_title(title)
    if save_name:
        plt.savefig(save_name)


# %%
def plot_model(model, title=None, save_name=None):
    # fig, ax = plt.subplots()
    source_emb = extract_emb(model, source_train_loader)
    target_emb = extract_emb(model, target_train_loader)

    print("Original embeddings of source / target", source_emb.shape, target_emb.shape)

    indexes = np.random.permutation(len(source_emb))[:1000]

    emb = np.concatenate((source_emb[indexes], target_emb[indexes]))
    domains = np.concatenate((np.ones((1000,)), np.zeros((1000,))))

    print("Samples embeddings", emb.shape, domains.shape)

    tsne = TSNE(n_components=2)

    emb_2d = tsne.fit_transform(emb)
    print("Dimension reduced embeddings", emb_2d.shape)
    plot_t_sne(emb_2d, domains, title=title, save_name=save_name)


# %% [markdown]
# # Pseudo labeling

# %%
from torch.utils.data import ConcatDataset


def train(model, loader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters())
    mu0, alpha, beta = 0.01, 10, 0.75
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: 0.01 / (1 + alpha * e / epochs) ** beta
    )

    total_batches = epochs * len(loader)
    with tqdm(total=total_batches, desc=f"Epoch 1/{epochs}", unit="batch") as pbar:
        for epoch in range(epochs):
            train_loss = 0.0

            for x, y in loader:
                x, y = x.cuda(), y.cuda()

                optimizer.zero_grad()
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                pbar.update(1)

            pbar.set_description(f"Epoch {epoch + 1}/{epochs}")
            scheduler.step()
            pbar.set_postfix(
                {
                    "train loss": round(train_loss / len(loader), 5),
                    "learning rate": optimizer.param_groups[0]["lr"],
                }
            )


def generate_pseudo_labels(model, dataset):
    model.eval()
    pseudo_labels = []
    for x, _ in dataset:
        with torch.no_grad():
            x = x.cuda()
            output = model(x)
            pseudo_label = output.argmax(dim=1)
            pseudo_labels.append(pseudo_label)
    return torch.cat(pseudo_labels).cpu().numpy()


wandb.init(entity="iksrawowip", name="Pseudo-labeling2")

for i in range(20):
    pseudo_labels = generate_pseudo_labels(naive_net, target_train_loader)
    mnistm_train_pseudo_label = MNISTM(mnist_m["x_train"], pseudo_labels, transform)
    combined_dataset = ConcatDataset([mnist_train, mnistm_train_pseudo_label])
    combined_loader = DataLoader(combined_dataset, batch_size=64)
    train(naive_net, combined_loader)
    test_acc, test_loss = eval_model(naive_net, target_test_loader)
    print(f"Test target loss: {test_loss}, test target acc: {test_acc}")
    wandb.log(
        {
            "Iteration": i,
            "Target domain test loss": test_loss,
            "Target domain test accuracy": test_acc,
        }
    )
    # plot_model(naive_net, save_name=f"figs_pseudo_labeling/t-SNE_{i}.pdf") # Need fix les figures, elle se stack au fur et à mesure
    # Faut juste add un plt.subplots()
    # Mais de toutes façon l'acc bouge pas donc ça serre à rien
wandb.finish()
