import numpy as np
import scipy.sparse as sp


SOLVER_OUTPUT = True
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
    import mosek

    if not SOLVER_OUTPUT:
        {mosek.iparam.log: 0}

except Exception, e:
    print "warning: cvxopt and or mosek not found"
    default_solver = 'gurobi'


def _solve_mosek(c, Aiq, biq, Aeq, beq, J):
    """
        Solve optimization problem
        min c' x
        s.t. Aiq x <= biq
             Aeq x == beq
             x[J] are integers
             x >= 0
        using the Mosek ILP solver
    """
    inf = 0.0  # for readability

    num_var = Aiq.shape[1]
    num_iq = Aiq.shape[0]
    num_eq = Aeq.shape[0]

    Aiq = Aiq.tocsr()
    Aeq = Aeq.tocsr()

    env = mosek.Env()
    task = env.Task(0, 0)
    task.putintparam(mosek.iparam.log, 0)

    task.appendvars(num_var)
    task.appendcons(num_iq + num_eq)

    for j in range(num_var):
        task.putcj(j, c[j])
        task.putbound(mosek.accmode.var, j, mosek.boundkey.lo, 0., +inf)

    for j in J:
        task.putvartype(j, mosek.variabletype.type_int)

    for i in range(num_iq):
        # Add inequality constraints
        start = Aiq.indptr[i]
        end = Aiq.indptr[i + 1]
        variables = Aiq.indices[start:end]
        coeff = Aiq.data[start:end]

        task.putarow(i, variables, coeff)
        task.putbound(mosek.accmode.con, i, mosek.boundkey.up, -inf, biq[i])

    for i in range(num_eq):
        # Add equality constraints
        start = Aeq.indptr[i]
        end = Aeq.indptr[i + 1]
        variables = Aiq.indices[start:end]
        coeff = Aiq.data[start:end]

        task.putarow(num_iq + i, variables, coeff)
        task.putbound(mosek.accmode.con, num_iq + i,
                      mosek.boundkey.fx, beq[i], +inf)

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


def _solve_gurobi(c, Aiq, biq, Aeq, beq, J):
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
    m.setParam(GRB.Param.OutputFlag, GUROBI_OUTPUT)

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


def solve_mip(c, Aiq, biq, Aeq, beq, J=None, solver=default_solver):
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
        return _solve_gurobi(c, Aiq, biq, Aeq, beq, J)
    elif solver == 'mosek':
        return _solve_mosek(c, Aiq, biq, Aeq, beq, J)


def _sparse_scipy_to_cvxopt(A):
    A_coo = A.tocoo()
    return spmatrix(A_coo.data.astype(float),
                    A_coo.row.astype(int),
                    A_coo.col.astype(int),
                    (A_coo.shape[0], A_coo.shape[1]), tc='d')
