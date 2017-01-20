import sys
import scipy.sparse as sp
import numpy as np

default_solver = 'gurobi'

# Try to import gurobi
try:
    from gurobipy import *
    TIME_LIMIT = 10 * 3600

except Exception, e:
    print "warning: gurobi not found"
    default_solver = 'mosek'

# Try to import mosek/cvxopt
try:
    import mosek

except Exception, e:
    print "warning: cvxopt and or mosek not found"
    default_solver = 'gurobi'


def _solve_mosek(c, Aiq, biq, Aeq, beq, J, output):
    """
        Solve optimization problem
        min c' x
        s.t. Aiq x <= biq
             Aeq x == beq
             x[J] are integers
             x >= 0
        using the Mosek ILP solver
    """
    def streamprinter(text):
        sys.stdout.write(text)
        sys.stdout.flush()

    inf = 0.0  # for readability

    num_var = Aiq.shape[1]
    num_iq = Aiq.shape[0]
    num_eq = Aeq.shape[0]

    env = mosek.Env()
    env.set_Stream(mosek.streamtype.log, streamprinter)

    task = env.Task(0, 0)
    task.set_Stream(mosek.streamtype.log, streamprinter)
    task.putintparam(mosek.iparam.log, 10 * output)

    task.appendvars(num_var)
    task.appendcons(num_iq + num_eq)

    # Coefficients
    task.putcslice(0, num_var, c)

    # Positivity
    task.putvarboundslice(0, num_var, [mosek.boundkey.lo] * num_var,
                                      [0.] * num_var,
                                      [+inf] * num_var)

    # Integers
    task.putvartypelist(J, [mosek.variabletype.type_int] * len(J))

    # Inequality constraints
    task.putaijlist(Aiq.row, Aiq.col, Aiq.data)
    task.putconboundslice(0, num_iq,
                          [mosek.boundkey.up] * num_iq,
                          [-inf] * num_iq,
                          biq)

    # Equality constraints
    task.putaijlist(num_iq + Aeq.row, Aeq.col, Aeq.data)
    task.putconboundslice(num_iq, num_iq + num_eq,
                          [mosek.boundkey.fx] * num_eq,
                          beq,
                          beq)

    task.putobjsense(mosek.objsense.minimize)
    task.optimize()

    sol = {}
    sol['x'] = np.zeros(num_var, float)

    if len(J) > 0:
        solsta = task.getsolsta(mosek.soltype.itg)
        task.getxx(mosek.soltype.itg, sol['x'])
    else:
        solsta = task.getsolsta(mosek.soltype.bas)
        task.getxx(mosek.soltype.bas, sol['x'])

    if solsta in [solsta.optimal,
                  solsta.near_optimal,
                  solsta.integer_optimal,
                  solsta.near_integer_optimal]:
        sol['status'] = 2
    elif solsta in [solsta.dual_infeas_cer,
                    solsta.near_dual_infeas_cer]:
        sol['status'] = 5
    elif solsta in [solsta.prim_infeas_cer,
                    solsta.near_prim_infeas_cer]:
        sol['status'] = 3
    elif solsta == solsta.unknown:
        sol['status'] = 1

    return sol


def solCallback(model, where):
    if where == GRB.callback.MIPSOL:
        solcnt = model.cbGet(GRB.callback.MIPSOL_SOLCNT)
        runtime = model.cbGet(GRB.callback.RUNTIME)
        if solcnt > 0 and runtime > TIME_LIMIT:
            model.terminate()


def _solve_gurobi(c, Aiq, biq, Aeq, beq, J, output):
    """
        Solve optimization problem
        min c' x
        s.t. Aiq x <= biq
             Aeq x == beq
             x[J] are integers
             x >= 0
        using the Gurobi solver
    """
    num_var = Aiq.shape[1]

    if J is None:
        J = range(num_var)

    Aiq = Aiq.tocsr()
    Aeq = Aeq.tocsr()
    J = set(J)

    m = Model()

    # Enable/disable output
    m.setParam(GRB.Param.OutputFlag, output)

    # Some solver parameters, see
    # http://www.gurobi.com/documentation/6.0/refman/mip_models.html
    m.setParam(GRB.Param.TimeLimit, TIME_LIMIT)
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
        end = Aiq.indptr[i + 1]
        variables = [x[j] for j in Aiq.indices[start:end]]
        coeff = Aiq.data[start:end]
        expr = gurobipy.LinExpr(coeff, variables)
        m.addConstr(lhs=expr, sense=gurobipy.GRB.LESS_EQUAL, rhs=biq[i])

    for i in range(Aeq.shape[0]):
        start = Aeq.indptr[i]
        end = Aeq.indptr[i + 1]
        variables = [x[j] for j in Aeq.indices[start:end]]
        coeff = Aeq.data[start:end]
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


def solve_mip(c, Aiq, biq, Aeq, beq, J=None, solver=default_solver, output=0):
    """
    Solve the ILP
        min c' x
        s.t. Aiq x <= biq
             Aeq x == beq
             x[J] are integers
             x >= 0
    using the solver `solver`.
    If `J` is not given, all variables are treated as integers.
    """
    if solver is None:
            solver = default_solver

    if J is None:
        J = range(Aiq.shape[1])

    if solver == 'gurobi':
        return _solve_gurobi(c, Aiq, biq, Aeq, beq, J, output)
    elif solver == 'mosek':
        return _solve_mosek(c, Aiq, biq, Aeq, beq, J, output)
