import pickle
import os
import dimod
import dimod.binary_quadratic_model
import dwave.inspector
import dwave.system
import minorminer
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave.preprocessing import ScaleComposite
import json

problems = ["nQueens", "rotatingRostering", "graphColoring"]

# customizable
problem = problems[2]


adv2pSampler = DWaveSampler(solver=dict(topology__type="zephyr"))

currDir = os.path.dirname(__file__)
problemDir = os.path.join(currDir, problem)
quboDir = os.path.join(problemDir, problem+"Qubo.pkl")
resultDir = os.path.join(problemDir,"annealResults")

with open(quboDir, 'rb') as f:
    qubo = pickle.load(f)
print("Qubo loaded.")
bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)

print("Trying to find embedding.")
embedding = minorminer.find_embedding(dimod.to_networkx_graph(bqm), adv2pSampler.to_networkx_graph())
if embedding == {}:
    print("No embedding found!")
    exit()
else:
    print("Embedding Found.")

cnt = 1
while(True):
    dirPath = os.path.join(resultDir,"run"+str(cnt))
    if(os.path.isdir(dirPath)):
        cnt += 1
    else:
        try:
            os.mkdir(dirPath)
            print("Directory " + "run"+str(cnt) + " to store result created.")
        except FileExistsError:
            print("Directory " + "run"+str(cnt) + " already exists.")
            exit()
        except PermissionError:
            print("Permission denied: Unable to create " + "run"+str(cnt) + " directory.")
            exit()
        except Exception as e:
            print(f"An error occured: {e}")
            exit()
        break 

print("Starting sampler.")
sampleset = FixedEmbeddingComposite(ScaleComposite(adv2pSampler), embedding=embedding).sample_qubo(
    qubo, 
    quadratic_range=adv2pSampler.properties["extended_j_range"],
    bias_range=adv2pSampler.properties["h_range"],
    num_reads=1000,
    auto_scale=True,
    label="Example - "+problem,
    #annealing_time=150
)
print("Sampler finished.")

pklDir = os.path.join(dirPath,"sampleset.pkl") 
with open(pklDir, "wb+") as f:
    pickle.dump(sampleset.to_serializable(), f)
print("Sampleset saved as pkl.")

try:
    jsonDir = os.path.join(dirPath,"sampleset.json")
    with open(jsonDir, 'w+') as f:
        json.dump(sampleset.to_serializable(),f,indent=6, skipkeys=True)
    print("Sampleset saved as json.")
except:
    print("Json serialization not possible. Skipping Json.")

csvDir = os.path.join(dirPath,"samples.csv")
pandaData = sampleset.to_pandas_dataframe()
pandaData.to_csv(csvDir, index=False)
print("Samples saved as csv.")

dwave.inspector.show(sampleset)

