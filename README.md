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

### Installing the Gurobi python interface

1. Download Gurobi Optimizer and request a license file. Place it in e.g. ```~/gurobi.lic```
2. Go to the installation path (OS X: ```/Library/gurobi650/mac64/```) and run ```python setup.py install```
3. Add an environment variable to where the license file is located: e.g. ```export GRB_LICENSE_FILE=~/gurobi.lic``` or  ```echo 'export GRB_LICENSE_FILE=~/gurobi.lic' >> ~/.bashrc```
4. Run a test: e.g. ```python /Library/gurobi650/mac64/examples/python/mip1.py```

For more help see the quick start guides at http://www.gurobi.com/documentation/.

## Run an example
```
python example_simple.py
python example_abstraction_lin2d.py
```

## Run code tests

```
python -m unittest discover
```

## Documentation

[See the sphinx documentation](doc/html).
