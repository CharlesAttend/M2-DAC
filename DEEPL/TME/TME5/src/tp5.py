import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *

#  TODO:


def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    mask = output != padcar
    # x.view(-1, x.size(2))
    # reduce none pour faire une moyenne après 🤔
    return CrossEntropyLoss(output * mask, target, reduce="none").mean()


class RNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        nonlinearity="tanh",
        batch_first=False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_first = batch_first

        self.f_x = nn.Linear(input_size, hidden_size, bias=False)
        self.f_h = nn.Linear(hidden_size, hidden_size)
        self.f_d = nn.Linear(hidden_size, output_size)

        if nonlinearity == "tanh":
            self.nonlinearity = nn.Tanh()
        elif nonlinearity == "relu":
            self.nonlinearity = nn.ReLU()

    def forward(self, x, h):
        """_summary_

        Parameters
        ----------
        x : (length, batch, input_size)
        h : (batch, hidden_size)

        Returns
        -------
        h_final : (length, batch, hidden_size)
        """
        h_final = torch.zeros((h.size(0), x.size(1), h.size(1)))
        # ic(h_final.size())
        if self.batch_first:
            for i in range(x.size(1)):
                h = self.one_step(x[:, i, :], h)
                # ic(h.size())
                h_final[:, i, :] = h
        else:
            for i in range(x.size(0)):
                h = self.one_step(x[i, :, :], h)
                h_final[i, :, :] = h
        return h_final

    def one_step(self, x, h=None):
        """

        Parameters
        ----------
        x :
            (batch,input_size)
        h :
            (batch,hidden_size)

        Returns
        -------
        h_t+1 : (batch,hidden_size)
        """
        if not h:
            h = torch.zeros(x.size(0), self.hidden_size)
        # ic(x.size())
        # ic(h.size())
        # ic(self.f_x(x).size())
        # ic(self.f_h(h).size())
        return self.nonlinearity(self.f_x(x) + self.f_h(h))

    def decode(self, h):
        return self.nonlinearity(self.f_d(h))


class LSTM(RNN):
    ...
    #  TODO:  Implémenter un LSTM


class GRU(nn.Module):
    ...
    #  TODO:  Implémenter un GRU


#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot
