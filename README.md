#  Quantum Annealing

This repository contains different examples of CSPs (Constraint Satisfaction Problems) and COPs (Constraint Optimization Problems) that can be run on the D-Wave quantum annealer. Each example has its own directory, with the data from one sample run already stored. Parts of the scripts that should be modified are marked with a *customizable* comment.

## Directory Structure

The root directory contains the [annealQubo.py](annealQubo.py) script, which is used to send problems to the annealer. The [displaySolutions.py](displaySolutions.py) script provides a custom printout for each problem solved with the quantum annealer.

Each problem has its own subdirectory. Within each subdirectory, there is a script to create the QUBO (Quadratic Unconstrained Binary Optimization) formular and the results from the annealer. Each annealer run gets its own directory. The returned sample set from the annealer is saved with all metadata as a complete object in a *.pkl* file, and the measurements of variables are saved in a *.csv* file.

## How to Use

The problem scripts (for example [nQueens.py](nQueens/nQueens.py)) can be run to create the QUBO needed for the annealer. The default QUBOs are already included in the repository, so this step is only necessary if the script is modified.

The [annealQubo.py](annealQubo.py) script contains one line that can be modified. The variable problem determines which problem gets sent to the annealer.

The [displaySolutions.py](displaySolutions.py) script has three variables that can be adjusted:

1.  **problem**: Selects the problem to display.
2.  **run**: Determines which run of the annealer is displayed.
3.  **maxSolutionsShown**: Limits the maximum number of samples shown if multiple samples have the same lowest energy.

## Python Versions Used

- For the QUBO creation scripts, Python 3.10 was used.
- For the annealing and display scripts, Python 3.12 was used.