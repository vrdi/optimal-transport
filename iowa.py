#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import os
from functools import partial
import json

import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from random import random

from gerrychain.metrics import efficiency_gap, mean_median
from gerrychain.updaters import cut_edges
from gerrychain.tree import recursive_tree_part
from gerrychain.grid import Grid
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election)
from gerrychain.proposals import recom, propose_random_flip, flip
from gerrychain.constraints import single_flip_contiguous

from gerrychain.accept import always_accept
from wasserplan import Pair

from sklearn.manifold import MDS



unique_label = "GEOID10"
pop_col = "TOTPOP"
district_col = "CD"



def num_splits(partition, df): # counting how many times a split is done to a county/block group because we don't want that
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

def MC_sample(jgraph, num_dist):
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


    initial_partition = Partition(jgraph, "CD", my_updaters) # by typing in "CD," we are saying to put every county into the congressional district that they belong to

    ideal_population = ideal_population = sum(initial_partition["population"].values()) / len(
        initial_partition
    )

    compactness_bound = constraints.UpperBound(
        lambda p: len(p["cut_edges"]), 2 * len(initial_partition["cut_edges"])
    )

    chain = MarkovChain(
        proposal=propose_random_flip,
        constraints=[single_flip_contiguous],
        accept=always_accept,
        initial_state=initial_partition,
        total_steps=20000
    )

    partitions=[] # recording partitions at each step
    for index, part in enumerate(chain):
        if index % 100 == 0:
            partitions += [part]

    return(partitions)



def build_distances_matrix(partitions):
    """
    :param partitions: list of partitions (plans) rom whoch to compute the distances matrix
    :returns: distances matrix
    """
    num = len(partitions)
    distances=np.zeros((num, num))
    for i in range(num):
        for j in range(i+1,num):
            distances[i][j] = Pair(partitions[i],partitions[j]).distance
            distances[j][i] = distances[i][j]
    return(distances)

        