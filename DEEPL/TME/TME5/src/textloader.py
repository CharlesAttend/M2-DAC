import sys
import unicodedata
import string
from typing import List
from torch.utils.data import Dataset, DataLoader
import torch
import re
from icecream import ic, install
install()

## Token de padding (BLANK)
PAD_IX = 0
## Token de fin de séquence
EOS_IX = 1

LETTRES = string.ascii_letters + string.punctuation + string.digits + " "
id2lettre = dict(zip(range(2, len(LETTRES) + 2), LETTRES))
id2lettre[PAD_IX] = "<PAD>"  ##NULL CHARACTER
id2lettre[EOS_IX] = "<EOS>"
lettre2id = dict(zip(id2lettre.values(), id2lettre.keys()))


def normalize(s):
    """enlève les accents et les caractères spéciaux"""
    return "".join(c for c in unicodedata.normalize("NFD", s) if c in LETTRES)


def string2code(s):
    """prend une séquence de lettres et renvoie la séquence d'entiers correspondantes"""
    return torch.tensor([lettre2id[c] for c in normalize(s)])


def code2string(t):
    """prend une séquence d'entiers et renvoie la séquence de lettres correspondantes"""
    if type(t) != list:
        t = t.tolist()
    return "".join(id2lettre[i] for i in t)


class TextDataset(Dataset):
    def __init__(self, text: str, *, maxsent=None, maxlen=None):
        """Dataset pour les tweets de Trump
        * fname : nom du fichier
        * maxsent : nombre maximum de phrases.
        * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        self.phrases = [
            re.sub(" +", " ", p[:maxlen]).strip() + "."
            for p in text.split(".")
            if len(re.sub(" +", " ", p[:maxlen]).strip()) > 0
        ]
        if maxsent is not None:
            self.phrases = self.phrases[:maxsent]
        self.maxlen = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, i):
        return string2code(self.phrases[i])


def pad_collate_fn(samples: List[torch.Tensor]):
    """
    Doit faire du padding
    Ajouter un eos à la fin de la phrase et padder le reste
    """
    maxlen = max([len(s) for s in samples])
    samples2 = []
    for s in samples:
        s = s.tolist()
        n = maxlen - len(s)
        s.append(EOS_IX)
        s = s + [PAD_IX] * n
        samples2.append(s)
    return torch.tensor(samples2).t()


if __name__ == "__main__":
    test = "C'est. Un. Test."
    ds = TextDataset(test)
    loader = DataLoader(ds, collate_fn=pad_collate_fn, batch_size=3)
    data = next(iter(loader))
    print(data)
    print("Chaîne à code : ", test)
    # Longueur maximum
    assert data.shape == (7, 3), data.shape
    print("Shape ok")
    # e dans les deux cas
    assert data[2, 0] == data[1, 2]
    print("encodage OK")
    # Token EOS présent
    assert data[5, 2] == EOS_IX
    print("Token EOS ok")
    # BLANK présent
    assert (data[4:, 1] == 0).sum() == data.shape[0] - 4
    print("Token BLANK ok")
    # les chaînes sont identiques
    s_decode = " ".join(
        [
            code2string(s).replace(id2lettre[PAD_IX], "").replace(id2lettre[EOS_IX], "")
            for s in data.t()
        ]
    )
    print("Chaîne décodée : ", s_decode)
    assert test == s_decode
    # " ".join([code2string(s).replace(id2lettre[PAD_IX],"").replace(id2lettre[EOS_IX],"") for s in data.t()])
