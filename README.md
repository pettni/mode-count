# mode-count

Python toolbox for abstraction and controller synthesis for mode-counting objectives

## Dependencies

Working Python environment, with
* numpy
* scipy
* matplotlib (for visualization)
* networkx
* One of the following ILP solvers
	* [Mosek](https://mosek.com), also requires [cvxopt](http://cvxopt.org)
	* [Gurobi](http://www.gurobi.com)

## Run an example
```
python example_simple.py
python example_abstraction_lin2d.py
```

## Run code tests

```
python -m unittest discover
```

## Further documentation

[See the sphinx documentation](doc/documentation.rst) (uncompiled)
