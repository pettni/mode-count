# mode-count

Python toolbox for abstraction and controller synthesis for mode-counting objectives

## Dependencies

Working Python environment, with
* numpy
* scipy
* matplotlib (for visualization)
* networkx
* mosek with Mosek Fusion for Python (tested with version 7.1.0.40) 

## Run an example
```
python example_simple.py
python example_abstraction_lin2d.py
```

## Run code tests

```
python -m unittest discover
```

## Code overview

Classes:
* Abstraction, represents an abstraction of a switched dynamical system
* CycleControl, computes mode-counting enforcing feedback controls

Main methods:
* lin_syst: returns a linear system description (A,B) of the aggregate dynamics on a graph
* synthesize: solve a mode-counting problem on a mode-graph
* simulate: simulate a synthesizes solution using matplotlib.animation
