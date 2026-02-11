import math
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from itertools import combinations
import networkx as nx
import random


def construct_graph(movies_fn, ratings_fn, min_rating=5, min_weight=10):
    """Construit le graphe des films:
    * movies_fn : movies.csv
    * ratings_fn : ratings.csv
    * min_rating : seuil minimal du score pour lier un utilisateur à un film
    * min_weight : seuil minimal du poids d'une arête pour la garder dans le graphe
    """
    movies = pd.read_csv(movies_fn)
    ratings = pd.read_csv(ratings_fn)

    rated_movies = ratings[ratings.rating >= min_rating]
    grouped_movies = rated_movies[["userId", "movieId"]].groupby("userId").agg(list)
    pair_freq = defaultdict(int)
    item_freq = defaultdict(int)

    for lst_movies in tqdm(grouped_movies["movieId"]):
        pairs = combinations(sorted(lst_movies), 2)
        for i in lst_movies:
            item_freq[i] += 1
        for i, j in pairs:
            pair_freq[(i, j)] += 1

    movies_graph = nx.Graph()
    log_total = math.log(sum(item_freq.values()))
    # Pointwise Mutual Information : pmi(x,y) = log p(x,y)/(p(x)p(y)) = log (p(x,y)) - log(p(x)) -log(p(y))
    for (i, j), f in pair_freq.items():
        pmi = f * (
            math.log(f) - math.log(item_freq[i]) - math.log(item_freq[j]) + log_total
        )
        if pmi >= min_weight:
            movies_graph.add_edge(i, j, weight=pmi)

    return movies_graph, movies


def random_walk(graph, num_walks=5, num_steps=10, p=1, q=1):
    """ "
    Construit un ensemble de chemins dans le graphe par marche aléatoire biaisée :
    * graph : graphe
    * num_walks: nombre de chemins par noeud
    * num_step : longueur des chemins
    * p : plus p est grand, plus l'exploration est incitée, p  petit -> plus il y a des retours en arriere
    * q : plus q est grand, plus la marche reste localisée, q petit -> s'écarte des noeuds explorés
    """

    def next_step(previous, current):
        def get_pq(n):
            if n == current:
                return p
            if graph.has_edge(n, previous):
                return 1
            return q

        weights = [w["weight"] / get_pq(n) for n, w in graph[current].items()]
        return random.choices(list(graph[current]), weights=weights)[0]

    walks = []
    nodes = list(graph.nodes())
    for walk_iter in range((num_walks)):
        for node in tqdm(nodes):
            walk = [node]
            cur_node = node
            prev_node = None
            for step in range(num_steps):
                next_node = next_step(prev_node, cur_node)
                walk.append(next_node)
                prev_node = cur_node
                cur_node = next_node
            walks.append(walk)
    return walks
