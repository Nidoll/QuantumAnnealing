from pyqubo import Binary
from pyqubo import Array
import numpy as np
import pickle

n = 5

gs = 2*n

vars = Array.create('v', shape=(gs,gs), vartype="BINARY")

# h - horizontal, v - vertical
orientations = np.ndarray((gs,gs),dtype=object)

for i in range(gs):
    for j in range(gs):
        if(((i+j)%2) == 0):
            orientations[i][j] = "v"
        else:
            orientations[i][j] = "h"

def getNeig(x,y):

    neigh = []
    neighClean = []

    hDir = [(0,1),(1,1),(1,0),(0,-1),(-1,-1),(-1,0)]
    vDir = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]
    if(orientations[x][y] == "h"):
        for (xd, yd) in hDir:
            neigh.append((x+xd,y+yd))
    else:
        for (xd, yd) in vDir:
            neigh.append((x+xd,y+yd))

    for (nx,ny) in neigh:
        if (nx >= 0 and nx <= (gs-1)):
            if (ny >= 0 and ny <= (gs-1)):
                if(not(neighClean.__contains__((nx,ny)))):
                    neighClean.append((nx,ny))

    return neighClean

edges = []
for i in range(gs):
    for j in range(gs):
        neighbars = getNeig(i,j)
        for n in neighbars:
            if not edges.__contains__(((i,j),n)) and not edges.__contains__((n,(i,j))):
                edges.append(((i,j),n))

H = 0*vars[0][0]

for i in range(gs):
    for j in range(gs):
        H += -1*vars[i][j]

for ((x1,y1),(x2,y2)) in edges:
    H += 2*vars[x1][y1]*vars[x2][y2]

# Tiles Vorgeben
#H += -1*vars[0][1]

# Block Felder
#f1 = (1,4)
#f2 = (7,6)
H += 2*vars[0][1] + 2*vars[0][2] + 2*vars[1][1] + 2*vars[1][2]
H += 2*vars[8][5] + 2*vars[8][6] + 2*vars[9][5] + 2*vars[9][6]

#def fieldToNeighbars(x,y):


model = H.compile()
qubo, offset = model.to_qubo()

#print(qubo)
#print((model.decode_sample({"v[0][0]":1,"v[3][0]":1,"v[0][3]":1,"v[1][2]":1,"v[2][1]":1,"v[3][3]":1},vartype="Binary")))

print("Qubo offset: ", offset)

with open("aztecDiamondQubo.pkl", "wb+") as f:
    pickle.dump(qubo, f)