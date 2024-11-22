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
from PIL import Image, ImageDraw

problems = ["nQueens", "rotatingRostering", "graphColoring"]

# customizable
problem = problems[2]
run = 1
maxSolutionsShown = 1

adv2pSampler = DWaveSampler(solver=dict(topology__type="zephyr"))

currDir = os.path.dirname(__file__)
problemDir = os.path.join(currDir, problem)
quboDir = os.path.join(problemDir, problem+"Qubo.pkl")
resultDir = os.path.join(problemDir,"annealResults")

with open(quboDir, 'rb') as f:
    qubo = pickle.load(f)
bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)

sampleDir = os.path.join(resultDir,"run"+str(run)+"/sampleset.pkl")
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
                val = mapping[np.nonzero(matrix[i,j] == 1)[0]]
            except:
                val = "error   |"
            print(val, end= " ")
        print("")
    print("")
    counter+=1

elif problem == "graphColoring":

    nodes = 0
    colors = 0

    counter = 1
    for s in sol:

        cnt = 0
        while True:
            try: 
                cnt += 1
                a = s["v["+str(cnt)+"][0]"]
            except:
                nodes = cnt
                break

        cnt = 0
        while True:
            try: 
                cnt += 1
                a = s["v[0]["+str(cnt)+"]"]
            except:
                colors = cnt
                break

        # display coloring
        points = np.array([(207,181),(392,233),(528,268),(673,134),(658,303),(777,392),(880,518),(783,544),(849,758),
                           (900,696),(708,653),(755,791),(613,833),(560,623),(494,503),(414,575),(338,501),(371,396),(582,440)],dtype=object)

        # last color is for errors
        colorsMapping = [(255,0,0),(255,255,0),(0,255,0),(0,255,255),(255,0,255),(0,0,255),(0,0,0)]

        img = Image.open("graphColoring/brandenburg.png")
        draw = ImageDraw.Draw(img)

        matrix = np.zeros((nodes,colors),dtype=int)

        for i in range(nodes):
            for j in range(colors):
                matrix[i,j] = s["v["+str(i)+"]["+str(j)+"]"]


        for i in range(nodes):
            try:
                color = np.nonzero(matrix[i] == 1)[0][0]
            except:
                color = 6
            draw.circle(points[i],25,fill=colorsMapping[color])

        img.show()


        