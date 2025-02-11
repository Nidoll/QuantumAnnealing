from pyqubo import Binary
from pyqubo import Array
import numpy as np
import pickle

n = 6
m = 10
r = 3

conn = [(0,1),(0,2),(0,3),(2,3),(2,5),(4,5)]

vars = Array.create('v', shape=(n,m), vartype="BINARY")

H = 0*vars[0][0]

# ring capacity
for i in range(m):
    h1 = Binary("h1"+"r"+str(i))
    h2 = Binary("h2"+"r"+str(i))

    x0 = vars[0][i]
    x1 = vars[1][i]
    x2 = vars[2][i]
    x3 = vars[3][i]
    x4 = vars[4][i]
    x5 = vars[5][i]

    H += (1*h1)
    H += (1*h2 + h1*h2 + x0*h1 + x1*h2 + x2*x3 + x2*x4 + x2*x5 + x3*x4 + x3*x5 + x4*x5 - x2*h1 - x2*h2 - x3*h1 - x3*h2 - x4*h1 - x4*h2 - x5*h1 - x5*h2)

# connections
for c in conn:
    (a,b) = c
    hs = []
    for i in range(m):
        h = Binary("hcr"+str(i)+"v"+str(a)+","+str(b))
        x1 = vars[a][i]
        x2 = vars[b][i]
        H += x1*x2 + 3*h - 2*x1*h - 2*x2*h
        hs.append(h)
    
    s1 = Binary("s1v"+str(a)+","+str(b)) 
    s2 = Binary("s2v"+str(a)+","+str(b)) 
    s3 = Binary("s3v"+str(a)+","+str(b)) 
    s4 = Binary("s4v"+str(a)+","+str(b))

    H += (sum(hs)-1*s1-2*s2-4*s3-8*s4-1)**2 

# minimze number of nodes
for i in range(n):
    for j in range(m):
        H += (1/(n*m+1))*vars[i][j]

model = H.compile()
qubo, offset = model.to_qubo()

print("Qubo offset: ", offset)

with open("sonetQubo.pkl", "wb+") as f:
    pickle.dump(qubo, f)