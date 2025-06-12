
import numpy as np


class UVCoupling:
	"""
	A data format for semi-analytic,
	interferometer mutual coupling parameters.
	"""        
	def __init__(self, X_mat, antpos, inverted=False):
		"""
		Parameters
		----------
		X_mat : ndarray
			Complex coupling parameter matrix, of shape
			(Npol, Nants, Nants, Ntimes, Nfreqs)
		antpos : dict
			Antenna position dictionary, keys
			are ant numbers, values are antenna
			positions in ENU [meters]. Assumed
			that the ordering in antpos matches
			the ordering of X_mat Nants dimension.
        inverted : bool, optional
            Whether the X_mat has been inverted or not.
            Default = False.
		"""
		self.X_mat = X_mat
		self.antpos = antpos
		self.ants = list(self.antpos.keys())
		self.Nants = len(self.ants)
		self.Npol, _Nants, _, self.Ntimes, self.Nfreqs = X_mat.shape
		assert self.Nants == _Nants
		assert self.Npol == 1, "Currently only supports single-pol"
        self.inverted = inverted

        # pre-compute Identity matrix
        self.I = np.zeros(1, self.Nants, self.Nants, 1, 1)
        self.I[:, range(self.Nants), range(self.Nants)] += 1

	def apply_to_data(self, data):
		raise NotImplementedError

	def to_hera_sim(self):
		raise NotImplementedError

	def write_npz(self):
		raise NotImplementedError

    def read_npz(self):
        raise NotImplementedError

    def invert(self, rcond=1e-15):
        raise NotImplementedError
        X_inv = np.zeros_like(self.X_mat)
        for p in range(self.Npols):
            for t in range(self.Ntimes):
                for f in range(self.Nfreqs):
                    X_inv[p, :, :, t, f] = np.linalg.pinv(self.X_mat[p, :, :, t, f], rcond=rcond)
        return UVCoupling(X_inv, self.antpos, inverted=True)

	@classmethod
	def from_npz(cls, ):
		raise NotImplementedError
		return cls(X_mat, antpos, **kwargs)


class CouplingInflate:
    """
    Take a redundantly-compressed coupling parameter vector
    and inflate it to an Nants x Nants coupling matrix,
    with antenna ordering set by antpos.
    """
    def __init__(self, coupling_vecs, antpos, redtol=1.0):
        """
        Parameters
        ----------
        coupling_vecs : ndarray
            Coupling term baseline vectors,
            shape (Nterms, 3) in ENU [meters]
        antpos : dict
            Antenna position dictionary
        redtol : float
            Redundancy tolerance [meters]
        """
        self.coupling_vecs = coupling_vecs
        self.antpos = antpos
        self.ants = list(antpos.keys())
        Nants = len(antpos)
        self.redtol = redtol
        self.shape = (Nants, Nants)

        idx = np.zeros(self.shape, dtype=np.int64)
        zeros = np.zeros(self.shape, dtype=np.bool)

        # iterate over antenna-pairs
        for i, ant1 in enumerate(self.ants):
            for j, ant2 in enumerate(self.ants):
                vec = antpos[ant1] - antpos[ant2]
                norm = np.linalg.norm(vecs - vec, axis=1)
                match = np.isclose(norm, 0, atol=redtol)
                if norm.any():
                    idx[i, j] = np.where(norm)[0]
                else:
                    zeros[i, j] = True

        self.idx = idx.ravel()
        self.zeros = zeros.ravel()

    def __call__(self, params):
        # params = (Npol, Nvec, Ntimes, Nfreqs)
        shape = params.shape[:2] + self.shape + params.shape[-2:]

        # params = (Npol, Nant^2, Ntimes, Nfreqs)
        params = params[:, :, self.idx]

        # zero-out missing vectors
        params[:, :, self.zeros] = 0.0

        # params = (Npol, Nant, Nant, Ntimes, Nfreqs)
        params = params.reshape(shape)

        return params


class RedCouplingAvg:
    """
    Take an Nants x Nants coupling matrix and compress
    down to redundant coupling vectors.
    """
    def __init__(self):
        raise NotImplementedError

    def __call__(self, params):
        pass


