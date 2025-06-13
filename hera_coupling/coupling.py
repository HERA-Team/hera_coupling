import copy
import numpy as np
from typing import Dict, Optional

from hera_cal.datacontainer import DataContainer


class UVCoupling:
    """
    A data format for semi-analytic mutual coupling parameters.
    
    This class stores coupling parameters and provides methods to apply them
    to visibility data, save/load from disk, and interface with herasim.
    """   
    def __init__(self, coupling: np.ndarray, antpos: Dict[int, np.ndarray], 
                 freqs: np.ndarray, times: Optional[np.ndarray] = None,
                 pols: Optional[list] = None, metadata: Optional[Dict] = None):
        """
        Parameters
        ----------
        coupling : ndarray
            Complex coupling parameters, of shape (Npol, Nants, Nants, Ntimes, Nfreqs)
        antpos : dict
            Antenna position dictionary, keys are ant numbers, values are antenna 
            positions in ENU [meters]. Ordering of ants in antpos sets the
            antenna ordering in the 'coupling' array.
        freqs : ndarray
            Frequency array in Hz. If None, uses indices.
        times : ndarray, optional  
            Time array in JD. If time axis is greater than 1, times must be provided.
        pols : list, optional
            Polarization strings (e.g., ['ee', 'nn']). If None, uses indices.
        metadata : dict, optional
            Additional metadata to store with coupling parameters. Could include information on
            the observation, instrument, or processing steps.
        """
        self.coupling = np.asarray(coupling)
        self.antpos = dict(antpos)  # Make a copy
        self.freqs = freqs
        self.times = times  
        self.metadata = metadata or {}
        # self.pols = pols or [f'pol_{i}' for i in range(self.coupling.shape[0])]

        # Derived properties
        self.ants = sorted(list(self.antpos.keys()))
        self.Nants = len(self.ants)
        
        # Validate shapes
        self._validate_shapes()
        
        # pre-compute the identity matrix along (Nants, Nants)
        self.I = np.zeros(1, self.Nants, self.Nants, 1, 1)
        self.I[:, range(self.Nants), range(self.Nants)] += 1

    def _validate_shapes(self):
        """Validate that all arrays have consistent shapes."""
        expected_shape = (len(self.pols), self.Nants, self.Nants, 
                         self.Ntimes, self.Nfreqs)
        
        if self.coupling.shape != expected_shape:
            raise ValueError(f"coupling shape {self.coupling.shape} doesn't match "
                           f"expected {expected_shape}")
                           
        if self.freqs is not None and len(self.freqs) != self.Nfreqs:
            raise ValueError(f"freqs length {len(self.freqs)} != Nfreqs {self.Nfreqs}")
            
        if self.times is not None and len(self.times) != self.Ntimes:
            raise ValueError(f"times length {len(self.times)} != Ntimes {self.Ntimes}")
        
    def _validate_data(self, data: DataContainer):
        """
        Validate that the input data is compatible with the coupling parameters.
        
        Parameters
        ----------
        data : DataContainer
            The visibility data to validate.
        
        Raises
        ------
        ValueError
            If the data does not match the expected shapes or dimensions.
        """
        ## TODO: support UVData object, and plain ndarray of shape (Nbltimes, Nfreqs)
        if data.Nants != self.Nants:
            raise ValueError(f"Data has {data.Nants} antennas, but coupling has {self.Nants} antennas.")
        
        if data.Nfreqs != self.coupling.shape[-1]:
            raise ValueError(f"Data has {data.Nfreqs} frequencies, but coupling has {self.coupling.shape[-1]} frequencies.")
        
        if data.Ntimes != self.coupling.shape[-2]:
            raise ValueError(f"Data has {data.Ntimes} times, but coupling has {self.coupling.shape[-2]} times.")
        
        if data.pols is not None and len(data.pols) != self.coupling.shape[0]:
            raise ValueError(f"Data has {len(data.pols)} polarizations, but coupling has {self.coupling.shape[0]} polarizations.")

    def invert(self, data: DataContainer, first_order: bool=False, multi_path: bool=False, inplace: bool=False):
        """
        """
        # TODO:
        # 1. Validate that the input data is compatible with the coupling parameters.
        # 2. Should have the same number of antennas, frequencies, and times.
        # 3. We may want to invert this once and store it, as such
        # we may want to have self.inverted = True/False and have this function
        # return UVCoupling(pinv(coupling), ...), and then have the

        # Copy the data if not inplace
        if not inplace:
            data = copy.deepcopy(data)


        for ti, fi in zip(data.times, data.freqs):
            for pi, pol in enumerate(data.pols):
                # Load the data container in workable format and apply the coupling parameters.
                data_matrix = to_matrix(
                    data, ti, fi, pol # TODO: Don't really like this
                )

                # Get coupling parameters
                X = self.coupling[pi, :, :, ti, fi]  # pi is the polarization index
        
                if first_order:
                    # Apply first-order coupling correction
                    # This has the form V1 = V0 + V0 X^\dagger + (V0 X^\dagger)^dagger
                    # which can be solved for using the the Stewart-Barlett algorithm
                    decoupled_vis = solve_stewart_barlett(X, data_matrix)

                else:
                    # Apply full coupling correction
                    # This would involve a more complex operation, possibly involving matrix inversion
                    coupling_matrix = np.eye(self.Nants) + X
                    if multi_path:
                        coupling_matrix = X.dot(X)

                    # Do this once 
                    pinv = np.linalg.pinv(coupling_matrix)

                    decoupled_vis = np.linalg.solve(coupling_matrix, data_matrix)

    def apply_coupling(self, data: DataContainer, first_order: bool=False, inplace: bool=False):
        """
        Apply the coupling parameters to the visibility data.
        TODO: Should we just use `hera_sim`?
        TODO: support UVData and ndarray

        Parameters
        ----------
        data : DataContainer
            The visibility data to which the coupling parameters will be applied.
        first_order : bool, optional
            If True, apply only first-order coupling terms. If False, use second order terms.
        
        Returns
        -------
        DataContainer
            The visibility data with coupling applied.
        """
        # TODO:
        # Validate that the input data is compatible with the coupling parameters.
        # Should have the same number of antennas, frequencies, and times.

        # Copy the data if not inplace
        if not inplace:
            data = copy.deepcopy(data)

        # Load the data container in workable format and apply the coupling parameters.
        data_matrix = to_matrix(
            data, ti, fi, pol # TODO: Don't really like this
        )
        
        if first_order:
            pass
        else:
            pass
        
        return data

    def write_npz(self):
        raise NotImplementedError

    @classmethod
    def from_npz(cls, ):
        # return cls(coupling, terms, antpos, **kwargs)
        raise NotImplementedError
    
def to_matrix(data: DataContainer):
    """
    Convert a DataContainer to N-antenna by N-antenna matrix format.
    This function extracts the data from a DataContainer and reshapes it into a 2D matrix
    where each row corresponds to an antenna and each column corresponds to another antenna.
    
    Parameters
    ----------
    data : DataContainer
        The data container to convert.
        
    Returns
    -------
    ndarray
        A 2D matrix representation of the data.
    """
    # Assuming data is a DataContainer with a 'data' attribute that is a 3D array
    raise NotImplementedError("to_matrix function is not implemented yet.")


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

    def __call__(self, coupling):
        # coupling = (Npol, Nvec, Ntimes, Nfreqs)
        shape = coupling.shape[:2] + self.shape + coupling.shape[-2:]

        # coupling = (Npol, Nant^2, Ntimes, Nfreqs)
        coupling = coupling[:, :, self.idx]

        # zero-out missing vectors
        coupling[:, :, self.zeros] = 0.0

        # coupling = (Npol, Nant, Nant, Ntimes, Nfreqs)
        coupling = coupling.reshape(shape)

        return coupling


class RedCouplingAvg:
    """
    Take an Nants x Nants coupling matrix and compress
    down to redundant coupling vectors.
    """
    def __init__(self):
        raise NotImplementedError

    def __call__(self, params):
        pass


