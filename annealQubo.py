import pickle
import os
import dimod
import dimod.binary_quadratic_model
import dwave.embedding
import dwave.inspector
import dwave.system
import matplotlib.pyplot as plt
import minorminer
import numpy as np
import networkx as nx
from dimod.generators import random_nae3sat
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave.preprocessing import ScaleComposite

with open("quboV2.pkl", "rb") as f:
    qubo = pickle.load(f)

with open("bqmV2.pkl", "rb") as f:
    bqm = pickle.load(f)

adv2pSampler = DWaveSampler(solver=dict(topology__type="zephyr"))
#adv2pSampler = DWaveSampler()

print("Searching embedding.")
embedding = minorminer.find_embedding(dimod.to_networkx_graph(bqm), adv2pSampler.to_networkx_graph())
print("Embedding Found.")

#sampleset = dwave.system.EmbeddingComposite(adv2pSampler).sample(bqm, num_reads=1000)

sampleset = FixedEmbeddingComposite(ScaleComposite(adv2pSampler), embedding=embedding).sample(
    bqm, 
    quadratic_range=adv2pSampler.properties["extended_j_range"],
    bias_range=adv2pSampler.properties["h_range"],
    num_reads=500,
    auto_scale=True,
    label="Example - nQueensV2",
    #annealing_time=150
)

pandaData = sampleset.to_pandas_dataframe()
pandaData.to_csv('nQueensTimeV2.csv', index=False)

dwave.inspector.show(sampleset)
