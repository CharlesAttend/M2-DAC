"""
Dif avec la copy du script: 
Fait une run wanb à chaque nouveau GRL factor : donc on a toutes les stats du training
"""

import math
import pickle

import torch
import torchvision
import wandb
from mnistm import create_mnistm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

create_mnistm()
mnist_pixels = torchvision.datasets.MNIST(".", train=True, download=True).data / 255
mean = mnist_pixels.mean().item()
std = mnist_pixels.std().item()

print(f"Mean {mean} and Std {std}")
mean = torch.tensor([mean, mean, mean])
std = torch.tensor([std, std, std])

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


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, factor=-1):
        ctx.save_for_backward(torch.tensor(factor))
        return x

    @staticmethod
    def backward(ctx, grad):
        (factor,) = ctx.saved_tensors

        reversed_grad = factor * grad

        return reversed_grad, None


class DANN(nn.Module):
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

        self.domain = nn.Sequential(
            nn.Linear(48 * 7 * 7, 100), nn.ReLU(inplace=True), nn.Linear(100, 1)
        )

    def forward(self, x, factor=1):
        x = self.cnn(x)
        x = x.flatten(start_dim=1)

        class_pred = self.classif(x)
        domain_pred = self.domain(GradientReversal.apply(x, -1 * factor))

        return class_pred, domain_pred


source_train_loader = DataLoader(mnist_train, batch_size=64)
target_train_loader = DataLoader(mnistm_train, batch_size=64)


def eval_dann(net, loader, source=True):
    net.eval()

    c_acc, d_acc, cls_loss, d_loss = 0, 0, 0.0, 0.0
    c = 0
    for x, y in loader:
        x = x.cuda()
        if source:
            d = torch.ones(len(x))
        else:
            d = torch.zeros(len(x))

        c += len(x)

        with torch.no_grad():
            cls_logits, domain_logits = net(x.cuda())
            cls_logits, domain_logits = cls_logits.cpu(), domain_logits.cpu()

        cls_loss += F.cross_entropy(cls_logits, y).item()
        d_loss += F.binary_cross_entropy_with_logits(domain_logits[:, 0], d).item()

        c_acc += (cls_logits.argmax(dim=1) == y).sum().item()
        d_acc += ((torch.sigmoid(domain_logits[:, 0]) > 0.5).float() == d).sum().item()

    return (
        round(100 * c_acc / c, 2),
        round(100 * d_acc / c, 2),
        round(cls_loss / len(loader), 5),
        round(d_loss / len(loader), 5),
    )


for factor in range(1, 11):
    epochs = 30

    dann = DANN().cuda()

    optimizer = torch.optim.SGD(dann.parameters(), lr=1.0, momentum=0.9)

    mu0, alpha, beta = 0.01, 10, 0.75
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: 0.01 / (1 + alpha * e / epochs) ** beta
    )

    def lmbd(e):
        return -1 + 2 / (
            1 + math.exp(-factor * e / (len(source_train_loader) * epochs))
        )

    wandb.init(
        entity="iksrawowip",
        name=f"GRL_f_{factor}",
        config={
            "epochs": epochs,
            "GRL_Factor": lmbd,
            "scheduler": scheduler,
            "mu0": mu0,
            "alpha": alpha,
            "beta": beta,
            "factor": factor,
        },
    )
    b = 0
    for epoch in range(epochs):
        cls_loss, domain_loss = 0.0, 0.0
        grl_factor = lmbd(b)
        print(f"GRL factor {grl_factor}")

        for (xs, ys), (xt, _) in zip(source_train_loader, target_train_loader):
            grl_factor = lmbd(b)
            b += 1

            xs, ys = xs.cuda(), ys.cuda()
            xt = xt.cuda()
            x = torch.cat((xs, xt))

            optimizer.zero_grad()
            cls_logits, domain_logits = dann(x, factor=grl_factor)

            ce = F.cross_entropy(cls_logits[: len(ys)], ys)

            preds = torch.cat((torch.ones(len(xs)), torch.zeros(len(xt)))).cuda()
            bce = F.binary_cross_entropy_with_logits(domain_logits[:, 0], preds)

            loss = ce + bce
            loss.backward()
            optimizer.step()

            cls_loss += ce.item()
            domain_loss += bce.item()

        c_acc_source, d_acc_source, c_loss_source, d_loss_source = eval_dann(
            dann, source_test_loader
        )
        print(
            f"[SOURCE] Class loss/acc: {c_loss_source} / {c_acc_source}%, Domain loss/acc: {d_loss_source} / {d_acc_source}%"
        )

        c_acc_target, d_acc_target, c_loss_target, d_loss_target = eval_dann(
            dann, target_test_loader, source=False
        )
        print(
            f"[TARGET] Class loss/acc: {c_loss_target} / {c_acc_target}%, Domain loss/acc: {d_loss_target} / {d_acc_target}%"
        )
        wandb.log(
            {
                "GRL Factor": grl_factor,
                "Class Accuracy on Source domain": c_acc_source,
                "Discriminator Accuracy on Source domain": d_acc_source,
                "Class loss on Source domain": c_loss_source,
                "Discriminator loss on Source domain": d_loss_source,
                "Class Accuracy on Target domain": c_acc_target,
                "Discriminator Accuracy on Target domain": d_acc_target,
                "Class Loss on Target domain": c_loss_target,
                "Discriminator Loss on Target domain": d_loss_target,
            }
        )

        cls_loss = round(cls_loss / len(source_train_loader), 5)
        domain_loss = round(domain_loss / (2 * len(source_train_loader)), 5)
        print(f"Epoch {epoch}, class loss: {cls_loss}, domain loss: {domain_loss}")
        scheduler.step()
    wandb.log(
        {
            "factor": factor,
            "Final GRL Factor": grl_factor,
            "Final Class Accuracy on Source domain": c_acc_source,
            "Final Discriminator Accuracy on Source domain": d_acc_source,
            "Final Class loss on Source domain": c_loss_source,
            "Final Discriminator loss on Source domain": d_loss_source,
            "Final Class Accuracy on Target domain": c_acc_target,
            "Final Discriminator Accuracy on Target domain": d_acc_target,
            "Final Class Loss on Target domain": c_loss_target,
            "Final Discriminator Loss on Target domain": d_loss_target,
        }
    )
    wandb.finish()
