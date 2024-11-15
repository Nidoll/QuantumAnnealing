import numpy as np
from pyqubo import Binary
from pyqubo import Array
import neal
from dwave.samplers import TreeDecompositionSolver
from dwave.samplers import SimulatedAnnealingSampler
import dimod
import pickle
import itertools as it

np.set_printoptions(edgeitems=30,linewidth=10000000000)

n = 5
days = 5
shifts = 3
smin = 1
smax = 3

# 7 days
shiftTable7 = np.array([[2,2,2,2,2,4,4],
                        [2,2,2,2,2,2,2],
                        [2,2,2,2,2,1,1],
                        [2,2,2,2,2,1,1]])

# 3 days
shiftTable3 = np.array([[2,2,4],
                        [2,2,2],
                        [2,2,1],
                        [2,2,1]])

# 4 days, 4 employe
shiftTable4small = np.array([[0,0,1,1],
                             [1,1,1,1],
                             [2,2,1,1],
                             [1,1,1,1]])

# 5 days, 4 employe
shiftTable5small = np.array([[0,0,0,1,1],
                             [1,1,1,1,1],
                             [2,2,2,1,1],
                             [1,1,1,1,1]])

# 5 days, 5 employe, 2 shifts
shiftTableOther = np.array([[1,1,2,2,2],
                            [2,2,2,1,1],
                            [2,2,1,2,2]])

if days == 7:
    shiftTable = shiftTable7
elif days == 3:
    if n == 8:
        shiftTable = shiftTable3
elif days == 4:
    shiftTable = shiftTable4small 
elif days == 5:
    if n == 4:
        shiftTable = shiftTable5small
        
if shifts==3:
    shiftTable = shiftTableOther

vars = Array.create('v', shape=(n,days,shifts), vartype="BINARY")
H = 0*vars[0,0,0]

# one hot encoding (check)
for i in range(n):
    for j in range(days):
            H += 2*(sum(vars[i,j])-1)**2

# shift requirements (check)
for i in range(days):
    for j in range(shifts):
        H += ((sum(vars[:,i,j])-shiftTable[j,i])**2)

# shift repetitions 
def valToInd(x):
    return (x//days,x%days)

if(smin > 1):
    #"""
    # minimum
    for i in range(n*days):
        (a,b) = valToInd(i)
        (c,d) = valToInd((i+1)%(n*days))
        (e,f) = valToInd((i-1)%(n*days))
        for j in range(shifts):
            H += vars[a,b,j] - vars[e,f,j]*vars[a,b,j] - vars[a,b,j]*vars[c,d,j] + vars[e,f,j]*vars[c,d,j]
    #"""

# maximum
for i in range(n*days):
    ind = np.zeros(smax+1,dtype=tuple)
    for j in range(smax+1):
        ind[j] = valToInd((i+j)%(n*days))
    for j in range(shifts):
        vlist = []
        for (a,b) in ind:
            vlist.append(vars[a,b,j])
        for (v1,v2) in list(it.product(vlist[:-1],[vlist[-1]])):
            H += 0.25*v1*v2

# weekend same shift
for i in range(n):
        for k in range(shifts):
            H += vars[i,days-1,k] + vars[i,days-2,k] - 2*vars[i,days-1,k]*vars[i,days-2,k]

# shift order
for i in range(n*days):
    (a,b) = valToInd(i%(n*days))
    (c,d) = valToInd((i+1)%(n*days))
    for j in range(shifts):
        if j == shifts-1:
            H += vars[a,b,j] - vars[c,d,j]*vars[a,b,j] - vars[a,b,j]*vars[c,d,0] + vars[c,d,j]*vars[c,d,0]
        elif j > 0:
            H += (vars[a,b,j] - vars[c,d,j]*vars[a,b,j] - vars[a,b,j]*vars[c,d,0] - vars[a,b,j]*vars[c,d,j+1] 
                + vars[c,d,j]*vars[c,d,0] + vars[c,d,j+1]*vars[c,d,0] + vars[c,d,j]*vars[c,d,j+1])

        
# enough free days
if True:
    for i in range(n*days):
        indList = []
        varList = []
        for j in range(2*days):
            indList.append(valToInd((i+j)%(n*days)))
        for (a,b) in indList:
            H += (-0.07143*vars[a,b,0])
            varList.append(vars[a,b,0])
    

model = H.compile()
qubo, offset = model.to_qubo()
bqm = model.to_bqm()
ising = model.to_ising()

with open("rotatingRosteringQubo.pkl", "wb+") as f:
    pickle.dump(qubo, f)

with open("rotatingRosteringBqm.pkl", "wb+") as f:
    pickle.dump(bqm, f)

with open("rotatingRosteringIsing.pkl", "wb+") as f:
    pickle.dump(ising, f)