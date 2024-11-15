import pickle
import os
import dimod
import dimod.binary_quadratic_model
import dwave.embedding
import dwave.inspector
import dwave.system
from dwave.system import DWaveSampler
import json
import numpy as np

problems = ["nQueens", "rotatingRostering"]

# customizable
problem = problems[1]
run = 1
maxSolutionsShown = 3

adv2pSampler = DWaveSampler(solver=dict(topology__type="zephyr"))

currDir = os.path.dirname(__file__)
problemDir = os.path.join(currDir, problem)
quboDir = os.path.join(problemDir, problem+"Qubo.pkl")
resultDir = os.path.join(problemDir,"annealResults")

with open(quboDir, 'rb') as f:
    qubo = pickle.load(f)
bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)

sampleDir = os.path.join(resultDir,"run1/sampleset.pkl")
with open(sampleDir, 'rb') as f:
    sampleset = dimod.SampleSet.from_serializable(pickle.load(f))

lowestSamples = sampleset.lowest()
sol = lowestSamples.samples(n=maxSolutionsShown)

dwave.inspector.show_bqm_sampleset(bqm, sampleset, adv2pSampler)

if problem == "nQueens":

    mapping = np.array(["X","Q"])
    cnt = 1
    for s in sol:
        print("Solution " + str(cnt) + ":")
        n = int(np.sqrt(len(s)))
        for i in range(n):
            for j in range(n):
                print(mapping[s["v["+str(i)+"]["+str(j)+"]"]],end=" ")
            print(" ")
        print("\n")
        cnt += 1

elif problem == "rotatingRostering":

    emps = 0
    days = 0
    shifts = 0

    counter = 1
    for s in sol:
        cnt = 0
        while True:
            try: 
                cnt += 1
                a = s["v["+str(cnt)+"][0][0]"]
            except:
                emps = cnt
                break
        
        cnt = 0
        while True:
            try: 
                cnt += 1
                s["v[0]["+str(cnt)+"][0]"]
            except:
                days = cnt
                break

        cnt = 0
        while True:
            try: 
                cnt += 1
                s["v[0][0]["+str(cnt)+"]"]
            except:
                shifts = cnt
                break

        break

    matrix = np.ndarray((emps,days,shifts), dtype=np.int32)
    for i in range(emps):
        for j in range(days):
            for k in range(shifts):
                try:
                    matrix[i,j,k] = s["v["+str(i)+"]["+str(j)+"]["+str(k)+"]"]
                except:
                    matrix[i,j,k] = 0

    mapping = np.array(["day off |", "early   |", "late    |", "night   |"])

    print("Solution " + str(counter)+":")
    for i in range(emps):
        print("Employe "+str(i)+": ", end=" ")
        for j in range(days):
            try:
                val = mapping[np.nonzero(matrix[i,j] == 1)[0]][0]
            except:
                val = "error   |"
            print(val, end= " ")
        print("")
    print("")
    counter+=1

        