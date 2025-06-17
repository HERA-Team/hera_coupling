import numpy as np
from scipy.linalg import schur, get_lapack_funcs


class SylvesterSolver:
    """
    TODO: Fix docstring
    
    A class to solve Sylvester equations of the form AX + XB = C. Essentially just a wrapper around scipy's `solve_sylvester` function
    that caches the Schur decomposition of A and B for efficiency.
    """
    def __init__(self, L, R):
        """
        Initialize the SylvesterSolver with matrices L and R.
        Parameters
        ----------
        L : ndarray
            The left matrix in the Sylvester equation.
        R : ndarray
            The right matrix in the Sylvester equation.
        """
        self.cache_schur_decomposition(L, R)

    def cache_schur_decomposition(self, L, R):
        """
        Initialize the SylvesterSolver with matrices L and R.
        Parameters
        ----------
        L : ndarray
            The left matrix in the Sylvester equation.
        R : ndarray
            The right matrix in the Sylvester equation.
        """
        # Schur decomposition of L
        T1, U1 = schur(L, output='real')  # SciPy uses 'real' even for complex
        # Schur of R^\dagger
        T2, U2 = schur(R.conj().T, output='real')
        self.T1, self.T2  = T1, T2
        self.U1, self.U2  = U1, U2
        self.U1H, self.U2H = U1.conj().T, U2.conj().T
        self.trsyl = get_lapack_funcs('trsyl', (T1, T2))

    def solve(self, C):
        """
        Solve the Sylvester equation AX + XB = C using the cached Schur decompositions.

        Parameters
        ----------
        C : ndarray
            The right-hand side matrix in the Sylvester equation.
        """
        F = self.U1H @ C @ self.U2 # transform RHS
        Y, scale, info = self.trsyl(self.T1, self.T2, F, tranb='C', isgn=1)
        if info != 0:
            raise RuntimeError(f"TRSYL failed (info={info})")
        Y *= scale # follow SciPy convention
        A = self.U1 @ Y @ self.U2H  # back-transform
        return 0.5 * (A + A.conj().T) # symmetrise against round-off
    
class SecondOrderSolver:
    """
    A class to solve second-order linear equations of the form AX = B.
    Essentially just a wrapper around scipy's `solve` function that caches the
    LU decomposition of A for efficiency.
    """
    def __init__(self, A):
        """
        Initialize the SecondOrderSolver with matrix A.
        
        Parameters
        ----------
        A : ndarray
            The matrix in the second-order equation.
        """
        self.cache_lu_decomposition(A)

    def cache_lu_decomposition(self, A):
        """
        Cache the LU decomposition of matrix A.
        
        Parameters
        ----------
        A : ndarray
            The matrix in the second-order equation.
        """
        self.lu, self.piv = get_lapack_funcs('getrf', (A,))  # LU decomposition

    def solve(self, B):
        """
        Solve the second-order equation AX = B using the cached LU decomposition.

        Parameters
        ----------
        B : ndarray
            The right-hand side matrix in the second-order equation.
        """
        X, info = self.lu(B, self.piv)
        if info != 0:
            raise RuntimeError(f"LU decomposition failed (info={info})")
        return X