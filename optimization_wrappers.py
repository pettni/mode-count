import numpy as np
import scipy.sparse

SOLVER_OUTPUT = False
default_solver = 'gurobi'

# Try to import gurobi
try:
    from gurobipy import * 
    TIME_LIMIT = 10 * 3600
    if SOLVER_OUTPUT:
            GUROBI_OUTPUT = 1
    else:
            GUROBI_OUTPUT = 0
except Exception, e:
    print "warning: gurobi not found"
    default_solver = 'mosek'

# Try to import mosek/cvxopt
try:
    from cvxopt import matrix, spmatrix, solvers
    import cvxopt.msk as msk
    import mosek

    if not SOLVER_OUTPUT:
        solvers.options['show_progress'] = False
        solvers.options['mosek'] = {mosek.iparam.log: 0} 

except Exception, e:
    print "warning: cvxopt and or mosek not found"
    default_solver = 'gurobi'

def _solve_mosek(c,Aiq,biq,Aeq,beq,J):
    solsta, x_out = msk.ilp( matrix(c), _sparse_scipy_to_cvxopt(Aiq), matrix(biq), _sparse_scipy_to_cvxopt(Aeq), matrix(beq), J)
    sol = {}
    sol['x'] = x_out
    sol['status'] = solsta
    return sol

def solCallback(model, where):
    if where == GRB.callback.MIPSOL:
        solcnt = model.cbGet(GRB.callback.MIPSOL_SOLCNT)
        runtime = model.cbGet(GRB.callback.RUNTIME)
        if solcnt > 0 and runtime > TIME_LIMIT:
            model.terminate()

def _solve_gurobi(c,Aiq,biq,Aeq,beq,J):

    # Warning: variables are lower bounded by 0!!

    num_var = Aiq.shape[1]

    if J == None:
        J = range(num_var)
        
    Aiq = Aiq.tocsr()
    Aeq = Aeq.tocsr()
    J = set(J)

    m = Model()

    # Enable/disable output
    m.setParam(GRB.Param.OutputFlag, GUROBI_OUTPUT)

    # Some solver parameters, see
    # http://www.gurobi.com/documentation/6.0/refman/mip_models.html
    m.setParam(GRB.Param.TimeLimit, TIME_LIMIT)   # 
    m.setParam(GRB.Param.MIPFocus, 1)       

    x = []
    for i in range(num_var):
        if i in J:
            x.append(m.addVar(vtype=GRB.INTEGER, obj=c[i]))
        else:
            x.append(m.addVar(obj=c[i]))
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

    sol = {}
    if m.status == gurobipy.GRB.status.OPTIMAL:
        sol['x'] = np.array([v.x for v in x])
        sol['primal objective'] = m.objVal
    sol['status'] = m.status

    return sol

def solve_lp(c,Aiq,biq,Aeq,beq, solver = default_solver):

        if solver == None:
                solver = default_solver
        
        if solver == 'gurobi':
            return _solve_gurobi(c,Aiq,biq,Aeq,beq,[])
        elif solver == 'mosek':
            return solvers.lp(matrix(c), _sparse_scipy_to_cvxopt(Aiq), matrix(biq), _sparse_scipy_to_cvxopt(Aeq), matrix(beq), 'mosek')

def solve_mip(c,Aiq,biq,Aeq,beq, J = None, solver = default_solver):

    if solver == None:
            solver = default_solver
        
    assert(Aiq.shape[1] == Aeq.shape[1])
    assert(Aiq.shape[0] == len(biq))
    assert(Aeq.shape[0] == len(beq))

    if J == None:
        J = range(Aiq.shape[1])

    if solver == 'gurobi':
        return _solve_gurobi(c,Aiq,biq,Aeq,beq,J)
    elif solver == 'mosek':
        return _solve_mosek(c,Aiq,biq,Aeq,beq,J)


def _sparse_scipy_to_mosek(A):
    A_coo = A.tocoo()
    return Matrix.sparse(A_coo.shape[0], A_coo.shape[1], list(A_coo.row.astype(int)), list(A_coo.col.astype(int)), list(A_coo.data.astype(float)))

def _sparse_scipy_to_cvxopt(A):
    A_coo = A.tocoo()
    return spmatrix(A_coo.data.astype(float), A_coo.row.astype(int), A_coo.col.astype(int), (A_coo.shape[0], A_coo.shape[1]))
