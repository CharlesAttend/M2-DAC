import torch
import torch.nn as nn
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def one_step(self, x, h):
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
        # ic(x.size())
        # ic(h.size())
        # ic(self.f_x(x).size())
        # ic(self.f_h(h).size())
        return self.nonlinearity(self.f_x(x) + self.f_h(h))

    def decode(self, h):
        return self.nonlinearity(self.f_d(h))


class SampleMetroDataset(Dataset):
    def __init__(self, data, length=20, stations_max=None):
        """
        * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
        * length : longueur des séquences d'exemple
        * stations_max : normalisation à appliquer
        """
        self.data, self.length = data, length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = (
            stations_max
            if stations_max is not None
            else torch.max(self.data.view(-1, self.data.size(2), self.data.size(3)), 0)[
                0
            ]
        )
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = (
            self.data.size(0),
            self.data.size(1),
            self.data.size(2),
        )

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.classes * self.nb_days * (self.nb_timeslots - self.length)

    def __getitem__(self, i):
        ## transformation de l'index 1d vers une indexation 3d
        ## renvoie une séquence de longueur length et l'id de la station.
        station = i // ((self.nb_timeslots - self.length) * self.nb_days)
        i = i % ((self.nb_timeslots - self.length) * self.nb_days)
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day, timeslot : (timeslot + self.length), station], station


class ForecastMetroDataset(Dataset):
    def __init__(self, data, length=20, stations_max=None):
        """
        * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
        * length : longueur des séquences d'exemple
        * stations_max : normalisation à appliquer
        """
        self.data, self.length = data, length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = (
            stations_max
            if stations_max is not None
            else torch.max(self.data.view(-1, self.data.size(2), self.data.size(3)), 0)[
                0
            ]
        )
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = (
            self.data.size(0),
            self.data.size(1),
            self.data.size(2),
        )

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.nb_days * (self.nb_timeslots - self.length)

    def __getitem__(self, i):
        ## Transformation de l'indexation 1d vers indexation 2d
        ## renvoie x[d,t:t+length-1,:,:], x[d,t+1:t+length,:,:]
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return (
            self.data[day, timeslot : (timeslot + self.length - 1)],
            self.data[day, (timeslot + 1) : (timeslot + self.length)]
        )
