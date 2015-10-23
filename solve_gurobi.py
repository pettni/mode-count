import numpy as np
import scipy.sparse

from gurobipy import *

TIME_LIMIT = 10 * 3600

def solCallback(model, where):
	if where == GRB.callback.MIPSOL:
		solcnt = model.cbGet(GRB.callback.MIPSOL_SOLCNT)
		runtime = model.cbGet(GRB.callback.RUNTIME)
		if solcnt > 0 and runtime > TIME_LIMIT:
			model.terminate()

def solve_mip(c,Aiq,biq,Aeq,beq, J = None):

	assert(Aiq.shape[1] == Aeq.shape[1])
	assert(Aiq.shape[0] == len(biq))
	assert(Aeq.shape[0] == len(beq))

	num_var = Aiq.shape[1]

	if J == None:
		J = range(num_var)
		
	Aiq = Aiq.tocsr()
	Aeq = Aeq.tocsr()
	J = set(J)

	m = Model()

	# Enable/disable output
	m.setParam(GRB.Param.OutputFlag, 1)

	# Some solver parameters, see
	# http://www.gurobi.com/documentation/6.0/refman/mip_models.html
	m.setParam(GRB.Param.TimeLimit, TIME_LIMIT)   # 
	m.setParam(GRB.Param.MIPFocus, 1)		

	x = []
	for i in range(num_var):
		if i in J:
			x.append(m.addVar(vtype=GRB.INTEGER, obj=c[i], lb=-gurobipy.GRB.INFINITY))
		else:
			x.append(m.addVar(obj=c[i], lb=-gurobipy.GRB.INFINITY))
	m.update()

	for i in range(Aiq.shape[0]):
		start = Aiq.indptr[i]
		end   = Aiq.indptr[i+1]
		variables = [x[j] for j in Aiq.indices[start:end]]
		coeff     = Aiq.data[start:end]
		expr = gurobipy.LinExpr(coeff, variables)
		m.addConstr(lhs=expr, sense=gurobipy.GRB.LESS_EQUAL, rhs=biq[i])

	for i in range(Aeq.shape[0]):
		start = Aeq.indptr[i]
		end   = Aeq.indptr[i+1]
		variables = [x[j] for j in Aeq.indices[start:end]]
		coeff     = Aeq.data[start:end]
		expr = gurobipy.LinExpr(coeff, variables)
		m.addConstr(lhs=expr, sense=gurobipy.GRB.EQUAL, rhs=beq[i])

   	m.update()
   	m.optimize(solCallback)
	# if m.status == gurobipy.GRB.status.OPTIMAL:
   	return np.array([v.x for v in x]), m.objVal
	# else:
	   	# print "solution status: ", m.status
		# return None


def main():
	c = np.array([-1,0], dtype=float)

	Aiq = scipy.sparse.coo_matrix( ((1,1), ((0,1), (0,1))), (2,2), dtype=float)
	biq = np.array([4,4])

	Aeq = scipy.sparse.coo_matrix( ((1,2), ((0,0), (0,1))), (1,2), dtype=float)
	beq = np.array([2])

	sol = solve_mip(c, Aiq, biq, Aeq, beq, [0,1])
	print sol 
if __name__ == '__main__':
	main()