import csv
import os
from functools import partial
import json

import geopandas as gp
import pandas as pd
import matplotlib.pyplot as plt

from gerrychain import (
   Election,
   Graph,
   MarkovChain,
   Partition,
   accept,
   constraints,
   updaters,
)
from gerrychain.metrics import efficiency_gap, mean_median
from gerrychain.proposals import recom
from gerrychain.proposals import flip
from gerrychain.updaters import cut_edges
from gerrychain.tree import recursive_tree_part

newdir = "./Outputs/"
os.makedirs(os.path.dirname(newdir + "init.txt"), exist_ok=True)
with open(newdir + "init.txt", "w") as f:
    f.write("Created Folder")
    
import numpy as np
from gerrychain import Partition
from gerrychain.grid import Grid
from wasserplan import Pair
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election)
from gerrychain.proposals import recom
from gerrychain.proposals import flip
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd
import geopandas as gpd
from gerrychain.tree import recursive_tree_part
import tqdm
from random import random
from gerrychain.metrics import efficiency_gap, mean_median

from gerrychain.updaters import cut_edges
from gerrychain import MarkovChain
from gerrychain.constraints import single_flip_contiguous
from gerrychain.proposals import propose_random_flip
from gerrychain.accept import always_accept

unique_label = "GEOID10"
pop_col = "TOTPOP"
district_col = "CD"

graph_path = "./IA_counties/IA_counties.shp"

graph = Graph.from_file(graph_path, reproject = False)

graph.to_json("ia_json.json")

jgraph = Graph.from_json("ia_json.json")

df = gpd.read_file(graph_path)

def num_splits(partition): # counting how many times a split is done to a county/block group because we don't want that
    df["current"] = df[unique_label].map(dict(partition.assignment))
    splits = sum(df.groupby("CD")["current"].nunique() > 1)
    return splits

def avg_pop_dist(partition):
    ideal_population = sum(partition["population"].values()) / len(
    partition
)
    total_deviation = sum([abs(v - ideal_population) for v in partition['population'].values()])
    return (total_deviation)/len(partition)
    
def pop_dist_pct(partition):
    ideal_population = ideal_population = sum(partition["population"].values()) / len(
    partition)
    total_deviation = total_deviation = sum([abs(v - ideal_population) for v in partition['population'].values()])
    avg_dist = total_deviation/len(partition)
    return avg_dist/ideal_population
    
    
def polsby_popper(partition):
#    print(partition["Area"])
    
    return (4*np.pi*partition["Area"])/np.square(partition["Perimeter"])
    
my_updaters = {
    "cut_edges": cut_edges,
    "population": updaters.Tally("TOTPOP", alias = "population"),
    "avg_pop_dist": avg_pop_dist,
    "pop_dist_pct" : pop_dist_pct,
    "area_land": updaters.Tally("ALAND10", alias = "area_land"),
    "area_water": updaters.Tally("AWATER10", alias = "area_water"),
    "Perimeter": updaters.Tally("perimeter", alias = "Perimeter"),
    "Area": updaters.Tally("area", alias = "Area")
}

num_elections = 3

election_names = [
    "PRES00",
    "PRES04",
    "PRES08",
]

election_columns = [
    ["PRES00D", "PRES00R"],
    ["PRES04D", "PRES04R"],
    ["PRES08D", "PRES08R"]
]

elections = [
    Election(
        election_names[i],
        {"Democratic": election_columns[i][0], "Republican": election_columns[i][1]},
    )
    for i in range(num_elections)
]

election_updaters = {election.name: election for election in elections}

my_updaters.update(election_updaters)

num_dist = 4

initial_partition = Partition(jgraph, "CD", my_updaters) # by typing in "CD," we are saying to put every county into the congressional district that they belong to

ideal_population = ideal_population = sum(initial_partition["population"].values()) / len(
    initial_partition
)

#proposal = partial(
#    flip, pop_col="TOTPOP", pop_target=ideal_population, epsilon=0.02, node_repeats=2
#)

compactness_bound = constraints.UpperBound(
    lambda p: len(p["cut_edges"]), 2 * len(initial_partition["cut_edges"])
)

chain = MarkovChain(
    proposal=propose_random_flip,
    constraints=[single_flip_contiguous],
    accept=always_accept,
    initial_state=initial_partition,
    total_steps=200
)

partitions=[] # recording partitions at each step
for part in chain:
    partitions += [part]
    
import wasserplan

distances=np.zeros((200,200))
for i in range(200):
    for j in range(i+1,200):
        distances[i][j] = wasserplan.Pair(partitions[i],partitions[j]).distance
        distances[j][i] = distances[i][j]
        
        
from sklearn.manifold import MDS

mds = MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=None, random_state=None, dissimilarity='euclidean')

pos=mds.fit(distances).embedding_

import matplotlib.pyplot as plt

X=[]
Y=[]
for i in range(200):
    X.append(pos[i][0])
    Y.append(pos[i][1])
  
plt.scatter(X,Y)

np.save("flip_run1.npy",distances)
