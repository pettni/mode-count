# Paper examples

The following examples appeared in *Petter Nilsson and Necmiye Ozay, Control synthesis for large collections of systems with mode-counting constraints, in Proceedings of the International Conference on Hybrid Systems: Computation and Control, 2016*. Scripts to produce the plots in the paper are provided.

Versions of relevant software used for examples in the paper
 * Python v. 2.7.10
 * [Gurobi](http://www.gurobi.com) v. 6.5.0
 * numpy v. 1.10.1
 * networkx v. 1.10

## Example 5.1: Numerical

Two-dimensional switched system with two linear modes, and an unsafe set around the origin.

```python
python example_5.1.py   	 # saves control strategy in 'example_5.1.save'
python example_5.1_plot.py   # reads 'example_5.1.save' and produces a plot
```

## Example 5.2: TCL's

Example for an aggregation of one-dimensional switched systems that is solved for two different desired mode-counts (high and low). Solved using the approach

1. Solve prefix-suffix LP
2. Round suffix solution to integer
3. Solve prefix ILP to find matching prefix

The desired mode count is entered in ```example_5.2.py``` on line 41. Before producing plots, the following script needs to be executed for both ```high``` and ```low``` desired mode counts.
```python
python example_5.2.py    		  # Compute abstraction, control strategy, and do simulation
```
Plots are produced as follows:
```python
python example_5.2_plot1.py 	  # Produces KDE plots (takes considerable time)
python example_5.2_plot2.py  	  # Produces time line plot
```