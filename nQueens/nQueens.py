import numpy as np
from pyqubo import Binary
from pyqubo import Array
import neal
from dwave.samplers import TreeDecompositionSolver
import pickle
import itertools as it

np.set_printoptions(edgeitems=30,linewidth=10000000000)

n = 5
vars = Array.create('v', shape=(n,n), vartype="BINARY")

H = 0*vars[0][0]

# rows
for k in range(n):
    for (i,j) in list(it.combinations(vars[k], 2)):
        H += 2*i*j

# colums
for k in range(n):
    for (i,j) in list(it.combinations(vars[:,k],2)):
        H += 2*i*j

# asscend diagonals
diagArr = np.empty(2*n-3, dtype=object)
diagArr[...] = [[] for _ in range(2*n-3)]
for i in range(n):
    for j in range(n):
        if i==0 and j==0:
            continue
        if i == n-1 and j==n-1:
            continue
        diagArr[i+j-1].append(vars[i][j])

for l in diagArr:
    for (i,j) in list(it.combinations(l,2)):
        H += 2*i*j

# descend diagonals
diagArr = np.empty(2*n-3, dtype=object)
diagArr[...] = [[] for _ in range(2*n-3)]
for i in range(n):
    for j in range(n):
        if i==0 and j==0:
            continue
        if i == n-1 and j==n-1:
            continue
        diagArr[(i)+(j-1)].append(vars[n-1-i][j])

for l in diagArr:
    for (i,j) in list(it.combinations(l,2)):
        H += 2*i*j

# negativ bias 
for i in range(n):
    for j in range(n):
        H += (-1)*vars[i,j]

model = H.compile()
qubo, offset = model.to_qubo()
bqm = model.to_bqm()
ising = model.to_ising()

with open("nQueensQubo.pkl", "wb+") as f:
    pickle.dump(qubo, f)

with open("nQueensBqm.pkl", "wb+") as f:
    pickle.dump(bqm, f)

with open("nQueensIsing.pkl", "wb+") as f:
    pickle.dump(ising, f)


