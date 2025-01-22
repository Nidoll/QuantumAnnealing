import numpy as np
from pyqubo import Binary
from pyqubo import Array
import neal
from dwave.samplers import TreeDecompositionSolver
from dwave.samplers import SimulatedAnnealingSampler
import dimod
import pickle
import itertools as it
import json

np.set_printoptions(edgeitems=30,linewidth=10000000000)

class rotatingRosteringC:
    def __init__(self, n=5, days=5, shifts=3, smin=1, smax=3, shiftTable=[],offset=0):
        self.n = n
        self.days = days
        self.shifts = shifts 
        self.smin = smin 
        self.smax = smax
        self.offset = offset

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=4, sort_keys=True)
    
    def fromJSON(self, data):
        self.__dict__ = data

n = 8
days = 7
shifts = 4
smin = 2
smax = 4

shiftTable = np.array([[2,2,2,2,2,4,4],
                        [2,2,2,2,2,2,2],
                        [2,2,2,2,2,1,1],
                        [2,2,2,2,2,1,1]])

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


# minimum
for i in range(n*days):
    (a,b) = valToInd(i)
    (c,d) = valToInd((i+1)%(n*days))
    (e,f) = valToInd((i-1)%(n*days))
    for j in range(shifts):
        xi = vars[a,b,j]
        xim = vars[e,f,j]
        xip = vars[c,d,j]
        q = xi - xim*xi + xim*xip - xi*xip

        H += q

# maximum
for i in range(n*days):
    indList = []
    for j in range(smax+1):
        indList.append(valToInd((i+j)%(n*days)))
    for j in range(shifts):
        (x1,y1) = indList[0]
        xi = vars[x1,y1,j]
        (x2,y2) = indList[1]
        xi1 = vars[x2,y2,j]
        (x3,y3) = indList[2]
        xi2 = vars[x3,y3,j]
        (x4,y4) = indList[3]
        xi3 = vars[x4,y4,j]
        (x5,y5) = indList[4]
        xi4 = vars[x5,y5,j]
        a1 = Binary("a"+str(i)+str(j)+"1")
        a2 = Binary("a"+str(i)+str(j)+"2")
        a3 = Binary("a"+str(i)+str(j)+"3")
        z1 = Binary("z"+str(i)+str(j)+"1")
        z2 = Binary("z"+str(i)+str(j)+"2")
        z3 = Binary("z"+str(i)+str(j)+"3")
        q1 = (xi + xi1 - 2*z1 - a1)**2
        q2 = (xi2 + xi3 -2*z2 - a2)**2
        q3 = (z1 + z2 - 2*z3 - a3)**2
        q4 = (z3*x4)
        H += (q1+q2+q3+q4)
            

# weekend same shift
for i in range(n):
        for k in range(shifts):
            H += vars[i,days-1,k] + vars[i,days-2,k] - 2*vars[i,days-1,k]*vars[i,days-2,k]

# shift order
for i in range(n*days):
    (a,b) = valToInd(i)
    (c,d) = valToInd((i+1)%(n*days))

    # shift 3
    x0 = vars[a,b,3]
    x1 = vars[c,d,0]
    x2 = vars[c,d,3]
    H += (x0 - x0*x1 - x0*x2 + 2*x1*x2)

    # shift 2
    x0 = vars[a,b,2]
    x1 = vars[c,d,0]
    x2 = vars[c,d,2]
    x3 = vars[c,d,3]
    H += (x0 - x0*x1 - x0*x2 - x0*x3 + 2*x1*x2 + 2*x1*x3 + 2*x2*x3)
        
# enough free days
for i in range(n*days):
    indList = []
    varList = []
    slackls = []
    for j in range(2*days):
        indList.append(valToInd((i+j)%(n*days)))
    for (a,b) in indList:
        varList.append(vars[a,b,0])
    s1 = Binary("s"+str(i)+"1")
    s2 = Binary("s"+str(i)+"2")
    s3 = Binary("s"+str(i)+"3")
    s4 = Binary("s"+str(i)+"4")
    #for j in range(2*days-2):
        #slackls.append(Binary("s"+str(i)+str(j)))
    #H += (sum(slackls)-sum(varList)+2)**2
    H += (1*s1 + 2*s2 + 4*s3 + 8*s4 - sum(varList) + 2)**2

model = H.compile()
qubo, offset = model.to_qubo()
bqm = model.to_bqm()
ising = model.to_ising()

print("Qubo offset:")
print(offset)

with open("rotatingRosteringQubo.pkl", "wb+") as f:
    pickle.dump(qubo, f)

with open("rotatingRosteringBqm.pkl", "wb+") as f:
    pickle.dump(bqm, f)

with open("rotatingRosteringIsing.pkl", "wb+") as f:
    pickle.dump(ising, f)

x = rotatingRosteringC(n=n,days=days,shifts=shifts,smin=smin,smax=smax,shiftTable=shiftTable,offset=offset)

with open("rotatingRosteringPara.json", 'w+') as f:
    f.write(x.toJSON())