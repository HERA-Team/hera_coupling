import copy
import numpy as np
from typing import Dict, Optional
from hera_cal.datacontainer import DataContainer

from .solvers import SylvesterSolver


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
        self.antpos = dict(antpos)
        self.freqs = freqs
        self.times = times  
        self.metadata = metadata or {}
        # self.pols = pols or [f'pol_{i}' for i in range(self.coupling.shape[0])]

        # Derived properties
        self.ants = sorted(list(self.antpos.keys()))
        self.Nants = len(self.ants)
        self.Ntimes = 1 if self.times is None else len(self.times)
        self.Nfreqs = self.freqs.size
        self.is_time_invariant = self.Ntimes == 1

        # Validate shapes
        self._validate_shapes()
        
        # pre-compute the identity matrix along (Nants, Nants)
        self.identity_matrix = np.zeros(1, self.Nants, self.Nants, 1, 1)
        self.identity_matrix[:, range(self.Nants), range(self.Nants)] += 1

        self.set_production(False)

    def _validate_shapes(self):
        """Validate that all arrays have consistent shapes."""
        if not self.production:
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
        if not self.production:
            ## TODO: support UVData object, and plain ndarray of shape (Nbltimes, Nfreqs)
            if data.Nants != self.Nants:
                raise ValueError(f"Data has {data.Nants} antennas, but coupling has {self.Nants} antennas.")
            
            if data.Nfreqs != self.coupling.shape[-1]:
                raise ValueError(f"Data has {data.Nfreqs} frequencies, but coupling has {self.coupling.shape[-1]} frequencies.")
            
            if data.Ntimes != self.coupling.shape[-2]:
                raise ValueError(f"Data has {data.Ntimes} times, but coupling has {self.coupling.shape[-2]} times.")
            
            if data.pols is not None and len(data.pols) != self.coupling.shape[0]:
                raise ValueError(f"Data has {len(data.pols)} polarizations, but coupling has {self.coupling.shape[0]} polarizations.")

    def set_production(self, production):
        """
        If running in production mode (no data validation)

        Parameters
        ----------
        production : bool
        """
        self.production = production

    def invert(self, data: DataContainer, first_order: bool=False, multi_path: bool=False, inplace: bool=False):
        """
        """
        # TODO:
        # 1. Validate that the input data is compatible with the coupling parameters.
        # 2. Should have the same number of antennas, frequencies, and times.
        # 3. We may want to invert this once and store it, as such
        # we may want to have self.inverted = True/False and have this function
        # return UVCoupling(pinv(coupling), ...), and then have the
        
        # Validate the input data
        self._validate_data(data)

        # Copy the data if not inplace
        if not inplace:
            data = copy.deepcopy(data)

        for fi in data.freqs:
            for pi, pol in enumerate(data.pols):
                # Pre-compute the Sylvester solver if time-invariant
                if self.is_time_invariant:
                    if first_order:
                        solver = SylvesterSolver(
                            self.coupling[pi, :, :, 0, fi],
                            np.eye(self.Nants) + self.coupling[pi, :, :, 0, fi].conj().T
                        )
                    else:
                        X = self.coupling[pi, :, :, 0, fi]
                        coupling_matrix = np.eye(self.Nants) + X
                        if multi_path:
                            coupling_matrix += X.dot(X)
                        
                        # Use the pseudo-inverse of the coupling matrix
                        inv_coupling_matrix = np.linalg.pinv(coupling_matrix)

                for ti in data.times:
                    # Load the data container in workable format and apply the coupling parameters.
                    data_matrix = to_matrix(
                        data, ti, fi, pol # TODO: Don't really like this
                    )

                    if first_order:
                        if not self.is_time_invariant:
                            # Get coupling parameters, pi is the polarization index
                            X = self.coupling[pi, :, :, ti, fi]

                            # Pre-compute the Sylvester solver for time-variant data
                            solver = SylvesterSolver(
                                X, np.eye(self.Nants) + X.conj().T
                            )
                        # Apply first-order coupling correction
                        # This has the form V1 = V0 + V0 X^\dagger + (V0 X^\dagger)^dagger
                        # which can be solved for using the the Stewart-Barlett algorithm
                        decoupled_vis = solver.solve(data_matrix)

                    else:
                        if not self.is_time_invariant:
                            # Get coupling parameters, pi is the polarization index
                            X = self.coupling[pi, :, :, ti, fi]
                            coupling_matrix = np.eye(self.Nants) + X
                            if multi_path:
                                coupling_matrix += X.dot(X)
                            inv_coupling_matrix = np.linalg.pinv(coupling_matrix)

                        # Apply full coupling correction
                        decoupled_vis = np.dot(
                            inv_coupling_matrix, data_matrix
                        ).dot(np.conj(inv_coupling_matrix.T))

                    # Store the decoupled visibility data back into the DataContainer
                    self.from_matrix(decoupled_vis, data, ti, fi, pol)

        return data

    def apply_coupling(self, data: DataContainer, first_order: bool=False, multi_path: bool=False, inplace: bool=False):
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
        multi_path : bool, optional
            If True, apply multi-path coupling corrections.
        inplace : bool, optional
            If True, modify the input data in place. If False, return a new DataContainer.

        Returns
        -------
        DataContainer
            The visibility data with coupling applied.
        """
        # TODO:
        # Validate that the input data is compatible with the coupling parameters.
        # Should have the same number of antennas, frequencies, and times.
        # TODO: should be able to deactivate these checks with self.production = True
        #   with self.set_production(True)
        self._validate_data(data)

        # Copy the data if not inplace
        if not inplace:
            data = copy.deepcopy(data)

        for fi in range(self.Nfreqs):
            for pi, pol in enumerate(data.pols):
                for ti in range(self.Ntimes):
                    # Load the data container in workable format and apply the coupling parameters.
                    data_matrix = self.to_matrix(data, ti, fi, pol)
                    X = self.coupling[pi, :, :, ti, fi]

                    if first_order:
                        coupled_vis = data_matrix + \
                            data_matrix.dot(X.conj().T) + \
                            (data_matrix.dot(X.conj().T)).conj().T
                    else:
                        coupling_matrix = np.eye(self.Nants) + X
                        if multi_path:
                            coupling_matrix += X.dot(X)

                        coupled_vis = np.dot(coupling_matrix, data_matrix).dot(np.conj(coupling_matrix.T))
        
                    self.from_matrix(coupled_vis, data, ti, fi, pol)
        
        return data
    
    def to_matrix(self, data: DataContainer, time_idx: int, freq_idx: int, 
                  pol: str) -> np.ndarray:
        """
        Convert DataContainer visibility data to matrix format.
        
        Parameters
        ----------
        data : DataContainer
            Visibility data
        time_idx : int
            Time index
        freq_idx : int 
            Frequency index
        pol : str
            Polarization string
            
        Returns
        -------
        np.ndarray
            Visibility matrix of shape (Nants, Nants)
        """
        # TODO: make this a module function, not method
        # TODO: support UVData and plain ndarray

        # Create antenna index mapping
        ant_to_idx = {ant: i for i, ant in enumerate(self.ants)}
        
        # Initialize visibility matrix
        vis_matrix = np.zeros((self.Nants, self.Nants), dtype=complex)
        
        # Fill matrix from data
        for bl in data.antpairs:
            if pol in data[bl]:
                ant1, ant2 = bl
                i, j = ant_to_idx[ant1], ant_to_idx[ant2]
                
                # Get visibility data
                vis_data = data[bl][pol]
                if vis_data.ndim >= 2:
                    vis_value = vis_data[time_idx, freq_idx] 
                else:
                    vis_value = vis_data[freq_idx]
                    
                vis_matrix[i, j] = vis_value
                if i != j:  # Add conjugate for off-diagonal
                    vis_matrix[j, i] = np.conj(vis_value)
                    
        return vis_matrix
    
    def from_matrix(self, vis_matrix: np.ndarray, data: DataContainer, 
                   time_idx: int, freq_idx: int, pol: str):
        """
        Store matrix visibility data back into DataContainer.
        
        Parameters
        ----------
        vis_matrix : np.ndarray
            Visibility matrix of shape (Nants, Nants)
        data : DataContainer
            DataContainer to modify
        time_idx : int
            Time index
        freq_idx : int
            Frequency index  
        pol : str
            Polarization string
        """
        # TODO: make this a module function, not a method
        # TODO: support UVData and ndarray
        ant_to_idx = {ant: i for i, ant in enumerate(self.ants)}
        
        for bl in data.bls:
            if pol in data[bl]:
                ant1, ant2 = bl
                i, j = ant_to_idx[ant1], ant_to_idx[ant2]
                
                # Store visibility data
                if data[bl][pol].ndim >= 2:
                    data[bl][pol][time_idx, freq_idx] = vis_matrix[i, j]
                else:
                    data[bl][pol][freq_idx] = vis_matrix[i, j]

    def write_npz(self):
        """
        TODO: Write the coupling parameters to an NPZ file.

        What are the necessary key-value pairs to store?
            - coupling: ndarray
            - Antenna names: ndarray
            - Antenna positions: dict
        """
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


def coupling_inflate(coupling, coupling_vecs, antpos, redtol=1.0):
    """
    Take a redundantly-compressed coupling parameter vector
    and inflate it to an Nants x Nants coupling matrix,
    with antenna ordering set by antpos.

    Parameters
    ----------
    coupling : ndarray
        Coupling parameter of shape (Npol, Nred, Ntimes, Nfreqs)
    coupling_vecs : ndarray
        Coupling term baseline vectors,
        shape (Nterms, 3) in ENU [meters]
    antpos : dict
        Antenna position dictionary
    redtol : float
        Redundancy tolerance [meters]

    Returns
    -------
    ndarray
    """
    return CouplingInflate(coupling_vecs, antpos, redtol=redtol)(coupling)


class CouplingInflate:
    """
    Take a redundantly-compressed coupling parameter vector
    and inflate it to an Nants x Nants coupling matrix,
    with antenna ordering set by antpos. Use this over
    the coupling_inflate() function for repeated calls.
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
        zeros = np.zeros(self.shape, dtype=bool)

        # iterate over antenna-pairs
        for i, ant1 in enumerate(self.ants):
            for j, ant2 in enumerate(self.ants):
                vec = antpos[ant1] - antpos[ant2]
                norm = np.linalg.norm(coupling_vecs - vec, axis=1)
                match = np.isclose(norm, 0, atol=redtol)
                if match.any():
                    idx[i, j] = np.where(match)[0]
                else:
                    zeros[i, j] = True

        self.idx = idx.ravel()
        self.zeros = zeros.ravel()

    def __call__(self, coupling):
        # coupling = (Npol, Nvec, Ntimes, Nfreqs)
        shape = coupling.shape[:1] + self.shape + coupling.shape[-2:]

        # coupling = (Npol, Nant^2, Ntimes, Nfreqs)
        coupling = coupling[:, self.idx]

        # zero-out missing vectors
        coupling[:, self.zeros] = 0.0

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


def expand_coupling(coupling, ants, new_ants):
    """
    Expand coupling parameter to a new set of antennas.

    Parameters
    ----------
    coupling : ndarray
        Coupling of shape (Npols, Nants, Nants, Ntimes, Nfreqs)
    ants : list
        List of antenna IDs for coupling
    new_ants : list
        Set of new antenna IDs to expand coupling to.

    Returns
    -------
    ndarray
    """
    # get all antennas and setup empty coupling output
    Nants = len(new_ants)

    # get indexing
    if isinstance(ants, np.ndarray):
        ants = ants.tolist()
    idx = [ants.index(a) if a in ants else -1 for a in new_ants]

    # expand to new coupling parameter
    coupling = coupling[:, idx][:, :, idx]

    # zero out missing antennas
    zero = np.where(np.array(idx) == -1)[0]
    coupling[:, zero] = 0
    coupling[:, :, zero] = 0

    return coupling


