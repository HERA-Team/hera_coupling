import numpy as np
from scipy.linalg import solve_sylvester
from hera_coupling.utils import SylvesterSolver

def test_sylvester_solver():
    nants = 5

    # Generate random matrices A and B
    A = np.random.normal(0, 1, size=(nants, nants)) + np.random.normal(0, 1, size=(nants, nants)) * 1j
    A = (A.T.conj() + A) / 2
    C = np.random.normal(0, 1, size=(nants, nants)) + 1j * np.random.normal(0, 1, size=(nants, nants))
    vc = A + np.dot(A, C.T.conj()) + np.conjugate(np.dot(A, C.T.conj())).T
    
    # Create an instance of SylvesterSolver
    solver = SylvesterSolver(C, np.eye(nants) + C.T.conj())
    solution = solver.solve(vc)

    # Solve the Sylvester equation directly using scipy
    expected_solution = solve_sylvester(C, np.eye(nants) + C.T.conj(), vc)

    # Check if the solutions are close
    assert np.allclose(solution, expected_solution, atol=1e-6), "The solutions do not match!"
    assert np.allclose(solution, A, atol=1e-6), "Solution does not match the input matrix"