import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import PiecewisePolynomial, MathematicalProgram, SnoptSolver

def quintic(a, t):
    return np.dot(a, np.array([1, t, t**2, t**3, t**4, t**5]))

def quintic_derivative(a, t):
    return np.dot(a, np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4]))

def quintic_dderivative(a, t):
    return np.dot(a, np.array([0, 0, 2, 6*t, 12*t**2, 20*t**3]))

def quintic_ddderivative(a, t):
    return np.dot(a, np.array([0, 0, 0, 6, 24*t, 60*t**2]))


def MakePiecewisePolynomial(coeffs, breaks) -> PiecewisePolynomial:
    from pydrake.polynomial import Polynomial
    return PiecewisePolynomial(
        [np.array([Polynomial(c) for c in coeff]) for coeff in coeffs],
        breaks,
    )


def solve_for_initial_state(rope_start_point, ref_x, l, rope_end_points=None):
    nq = len(ref_x.flatten())
    prog = MathematicalProgram()

    q = prog.NewContinuousVariables(nq)

    p = q[:3]-rope_start_point
    prog.AddQuadraticConstraint(p[0]**2 +p[1]**2 +p[2]**2  - l**2, 0, 0)
    for i in range(len(ref_x)//3-1):
        p = q[i*3:(i+1)*3]-q[(i+1)*3:(i+2)*3]
        prog.AddQuadraticConstraint(p[0]**2 +p[1]**2 +p[2]**2  - l**2, 0, 0)
    
    if rope_end_points is not None:
        p = q[-3:]-rope_end_points
        prog.AddQuadraticConstraint(p[0]**2 +p[1]**2 +p[2]**2  - l**2, 0, 0)
        
    
    prog.AddQuadraticErrorCost(np.eye(len(ref_x)), ref_x, q)
    
    prog.SetInitialGuess(q, ref_x)

    solver = SnoptSolver()

    result = solver.Solve(prog)
    return result.GetSolution(q)
