from utils import random_walk,construct_graph
import math
from tqdm import tqdm
import networkx as nx
from torch import nn
from torch.utils.data import DataLoader
import random
import torch
from torch.utils.tensorboard import SummaryWriter

import time
import logging

logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##  TODO: 

if __name__=="__main__":
    PATH = "./ml-latest-small/"
    logging.info("Constructing graph")
    movies_graph, movies = construct_graph(PATH + "movies.csv", PATH + "ratings.csv")
    logging.info("Sampling walks")
    walks = random_walk(movies_graph,5,10,1,1)
    nodes2id = dict(zip(movies_graph.nodes(),range(len(movies_graph.nodes()))))
    id2nodes = list(movies_graph.nodes())
    id2title = [movies[movies.movieId==idx].iloc[0].title for idx in id2nodes]
    nx.Graph()