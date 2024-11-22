import numpy as np
from pyqubo import Binary
from pyqubo import Array
from dwave.samplers import TreeDecompositionSolver
import pickle
import itertools as it
from PIL import Image, ImageDraw

np.set_printoptions(edgeitems=30,linewidth=10000000000)

mapNameNum = {"Prignitz":0,"Ostpriegnitz-Ruppin":1,"Oberhavel":2,"Uckermark":3,"Barnim":4,"Maerkisch-Oderland":5,"Frankfuhrt-Oder":6,
              "Oder-Spree":7,"Cottbus":8,"Spree-Neisse":9,"Dahme-Spreewald":10,"Oderspreewald-Lausitz":11,"Elbe-Elster":12,
              "Teltow-Flaeming":13,"Potsdam":14,"Potsda-Mittelmark":15,"Brandenburg an der Havel":16,"Havelland":17,"Berlin":18}

mapNumName = np.empty(19,dtype=object)

cnt = 0
for k in mapNameNum:
    mapNumName[cnt] = k
    cnt+=1    

adjacenceList = [[1],
                 [0,2,17],
                 [1,3,4,18],
                 [2,4],
                 [2,3,5,18],
                 [18,6,7],
                 [5,7],
                 [5,6,9,10,18],
                 [9],
                 [8,7,10,11],
                 [18,7,9,11,12,13],
                 [9,10,12],
                 [11,10,13],
                 [12,10,18,15],
                 [18,15,17],
                 [13,14,16,17],
                 [15,17],
                 [1,2,18,14,16],
                 [2,4,5,7,10,13,15,14,17]]

adjacenceList = np.array(adjacenceList,dtype=object)

# customizable
colors = 3
berlin = False


if(berlin):
    nodes = 19
else:
    nodes = 18

vars = Array.create('v', shape=(nodes,colors), vartype="BINARY")

H = 0*vars[0][0]

# one hot encoding
for i in range(nodes):
    H += (sum(vars[i])-1)**2

# border constraint
for i in range(nodes):
    for neighbar in adjacenceList[i]:
        if berlin:
            for j in range(colors):
                H += vars[i,j]*vars[neighbar,j]
        else:
            if neighbar == 18:
                continue
            else:
                for j in range(colors):
                    H += vars[i,j]*vars[neighbar,j]

model = H.compile()
qubo, offset = model.to_qubo()
bqm = model.to_bqm()
ising = model.to_ising()

with open("graphColoringQubo.pkl", "wb+") as f:
    pickle.dump(qubo, f)

with open("graphColoringBqm.pkl", "wb+") as f:
    pickle.dump(bqm, f)

with open("graphColoringIsing.pkl", "wb+") as f:
    pickle.dump(ising, f)




