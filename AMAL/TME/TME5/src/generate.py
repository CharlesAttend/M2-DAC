from textloader import string2code, id2lettre
import math
import torch

#  TODO:  Ce fichier contient les différentes fonction de génération


def generate(rnn, emb, decoder, eos, start="", maxlen=200):
    """Fonction de génération (l'embedding et le decodeur être des fonctions du rnn). Initialise le réseau avec start (ou à 0 si start est vide) et génère une séquence de longueur maximale 200 ou qui s'arrête quand eos est généré.
    * rnn : le réseau
    * emb : la couche d'embedding
    * decoder : le décodeur
    * eos : ID du token end of sequence
    * start : début de la phrase
    * maxlen : longueur maximale
    """
"""    for i in range(maxlen):
        generated = [torch.tensor(torch.randint(len(lettre2id), (1,))).to(device)]
        for i in range(maxlen):
            embdeded = emb(nn.Embeddings())
            encoded = rnn(embdeded)
            decoded = decoder(encoded)
            generated.append(model.decode(h))
        generated = torch.stack(generated[1:])
        print("".join([id2lettre[int(i)] for i in generated.squeeze()]))"""
    l = 0
	res = start
	output = start
	while l < maxlen:
		output = decoder(rnn(emb(output)).argmax(axis=1))
		if output == eos:
			break
		res += output
		l+=1
	return start
    #  TODO:  Implémentez la génération à partir du RNN, et d'une fonction decoder qui renvoie les logits (logarithme de probabilité à une constante près, i.e. ce qui vient avant le softmax) des différentes sorties possibles


def generate_beam(rnn, emb, decoder, eos, k, start="", maxlen=200):
    """
    Génere une séquence en beam-search : à chaque itération, on explore pour chaque candidat les k symboles les plus probables; puis seuls les k meilleurs candidats de l'ensemble des séquences générées sont conservés (au sens de la vraisemblance) pour l'itération suivante.
    * rnn : le réseau
    * emb : la couche d'embedding
    * decoder : le décodeur
    * eos : ID du token end of sequence
    * k : le paramètre du beam-search
    * start : début de la phrase
    * maxlen : longueur maximale
    """
    #  TODO:  Implémentez le beam Search


# p_nucleus
def p_nucleus(decoder, alpha: float):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        * decoder: renvoie les logits étant donné l'état du RNN
        * alpha (float): masse de probabilité à couvrir
    """

    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
           * h (torch.Tensor): L'état à décoder
        """
        #  TODO:  Implémentez le Nucleus sampling ici (pour un état s)

    return compute
