import copy
import numpy as np
from typing import Dict, Optional, Union, List, Tuple
from hera_cal.datacontainer import DataContainer

class UVCoupling:
    """
    A data format for semi-analytic mutual coupling parameters.
    
    This class stores coupling parameters and provides methods to apply them
    to visibility data, save/load from disk, and interface with hera_sim.
    """   
    def __init__(
            self,
            coupling: np.ndarray,
            antpos: Dict[int, np.ndarray],
            freqs: np.ndarray,
            times: Optional[np.ndarray] = None,
            pols: Optional[List[str]] = None,
            metadata: Optional[Dict] = None,
            first_order: bool = False,
            inverted: bool = False
        ):
        """
        Initialize UVCoupling container.
        
        Parameters
        ----------
        coupling : np.ndarray
            Complex coupling parameters with shape (Npol, Nants, Nants, Ntimes, Nfreqs)
        antpos : Dict[int, np.ndarray]
            Antenna positions in ENU coordinates [meters]. Keys are antenna numbers.
        freqs : np.ndarray
            Frequency array in Hz.
        times : np.ndarray, optional
            Time array in JD. Required if Ntimes > 1.
        pols : List[str], optional
            Polarization strings (e.g., ['ee', 'nn']).
        metadata : Dict, optional
            Additional metadata dictionary.
        first_order : bool, optional
            Whether to treat coupling as first-order only.
        """
        self.coupling = coupling
        self.antpos = dict(antpos)
        self.freqs = freqs
        self.times = times  
        self.metadata = metadata or {}
        self.pols = pols

        # Derived properties
        self.ants = sorted(list(self.antpos.keys()))
        self.Nants = len(self.ants)
        self.Ntimes = 1 if self.times is None else len(self.times)
        self.Nfreqs = self.freqs.size

        # Check if coupling is time-invariant
        self._is_time_invariant = (coupling.shape[-2] == 1)  # Default to time-invariant
        self._inverted = inverted
        self._inversion_cache = {}  # Cache for inverted coupling if needed

        # Validate shapes
        self._validate_shapes()
        
        # pre-compute the identity matrix along (Nants, Nants)
        self.identity_matrix = np.zeros(1, self.Nants, self.Nants, 1, 1)
        self.identity_matrix[:, range(self.Nants), range(self.Nants)] += 1

        # Create antenna index mapping
        self.ant_to_idx = {ant: i for i, ant in enumerate(self.ants)}
        self.idx_to_ant = {i: ant for ant, i in self.ant_to_idx.items()}

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
            
    @property
    def is_inverted(self) -> bool:
        """Check if the coupling operator is currently inverted."""
        return self._inverted
    
    @property
    def is_time_invariant(self) -> bool:
        """Check if the coupling operator is time-invariant."""
        return self._is_time_invariant
        
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the coupling matrix."""
        return self.coupling.shape

    def set_production(self, production):
        """
        If running in production mode (no data validation)

        Parameters
        ----------
        production : bool
        """
        self.production = production

    def invert(self, first_order, multi_path) -> None:
        """
        """
        # TODO:
        # 1. We may want to invert this once and store it, as such
        #    we may want to have self.inverted = True/False and have this function
        #    return UVCoupling(pinv(coupling), ...), and then have the
        # 
        # TODO: we shoudl probably cache based on the assumption of first order / multi-path
        if self.first_order:
            # For first-order, just toggle the state
            self._inverted = not self._inverted
        else:
            # For higher-order, compute matrix inverse
            self.coupling = np.linalg.pinv(self.coupling)
            self._inverted = not self._inverted

    def apply_coupling(self, data: DataContainer | np.ndarray, first_order: bool=False, multi_path: bool=False, inplace: bool=False):
        """
        Apply the coupling parameters to the visibility data.
        TODO: Should we just use `hera_sim`?
        TODO: support UVData and ndarray

        Parameters
        ----------
        data : DataContainer or np.ndarray
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

                    # Get the coupling parameters for the current polarization and frequency
                    coupled_vis = self.coupling.apply_coupling(pol, fi, ti)

                    self.from_matrix(coupled_vis, data, ti, fi, pol)
        
        return data
    
    def write(self, filename: str, **kwargs) -> None:
        """
        TODO: Write the coupling parameters to an NPZ file.

        What are the necessary key-value pairs to store?
            - coupling: ndarray
            - Antenna names: ndarray
            - Antenna positions: dict
            - Frequencies: ndarray
            - Times: ndarray (if not time-invariant)
        """
        raise NotImplementedError

    @classmethod
    def read(cls, filename: str, **kwargs) -> "UVCoupling":
        """
        Read coupling parameters from a file and return a UVCoupling object.

        Parameters
        ----------
        filename : str
            Path to the file containing coupling parameters.
        **kwargs : dict
            Additional keyword arguments to pass to the constructor.

        Returns
        -------
        UVCoupling
            An instance of UVCoupling with the loaded parameters.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If the file does not contain valid coupling parameters.
        """
        data = load(filename)

        uvcoupling = cls(
            coupling=data['coupling'],
            freqs=data['freqs'],
            times=data.get('times', None),
            pols=data.get('pols', None),
            metadata=data.get('metadata', {}),
            first_order=data.get('first_order', False), 
            antpos=data['antpos'], 
            **kwargs
        )
        raise uvcoupling
    
def _extract_data_matrix(
    data: DataContainer, 
    time_idx: int, 
    freq_idx: int, 
    pol: str
) -> np.ndarray:
    """
    Extract visibility matrix from DataContainer.
    
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
    ant_to_idx = {ant: i for i, ant in enumerate(ants)}
    
    # Initialize visibility matrix
    vis_matrix = np.zeros((Nants, Nants), dtype=complex)
    
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

def _insert_data_matrix(
    vis_matrix: np.ndarray, 
    data: DataContainer, 
    time_idx: int, 
    freq_idx: int, 
    pol: str
) -> None:
    """
    Insert matrix visibility data back into DataContainer.
    
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
    # TODO: support UVData and ndarray
    for bl in data.bls:
        if pol in data[bl]:
            ant1, ant2 = bl
            i, j = ant_to_idx[ant1], ant_to_idx[ant2]
            
            # Store visibility data
            if data[bl][pol].ndim >= 2:
                data[bl][pol][time_idx, freq_idx] = vis_matrix[i, j]
            else:
                data[bl][pol][freq_idx] = vis_matrix[i, j]


def _apply_coupling_forward(
    data,
    uvcoupling: UVCoupling,
    first_order: bool = False,
    multi_path: bool = False,
    inplace: bool = False
):
    """
    Apply coupling parameters to visibility data in forward mode.
    """
    # Build an identity matrix for the number of antennas
    nants = uvcoupling.Nants
    identity = np.eye(nants, dtype=np.complex128)

    for ti in range(uvcoupling.Ntimes):
        for fi in range(uvcoupling.Nfreqs):
            for pi, pol in enumerate(uvcoupling.pols):
                # Extract the visibility matrix for the current time, frequency, and polarization
                vis_matrix = _extract_data_matrix(uvcoupling, data, ti, fi, pol)
                coupling_matrix = uvcoupling.coupling[pi, :, :, ti, fi]

                if multi_path and not first_order:
                    # If multi-path coupling is enabled, we need to dot the coupling matrix with itself
                    coupling_matrix += coupling_matrix.dot(coupling_matrix)


                # Apply the coupling parameters
                if first_order:
                    # Apply first-order coupling
                    coupled_vis = (
                        vis_matrix +
                        np.dot(vis_matrix, coupling_matrix.T.conj()) + 
                        np.dot(vis_matrix, coupling_matrix.T.conj()).T.conj()
                    )
                    
                else:
                    # Apply second-order or higher coupling
                    coupling_matrix += identity
                    coupled_vis = np.dot(
                        np.dot(coupling_matrix, vis_matrix),
                        coupling_matrix.T.conj()
                    )

                # Insert the modified visibility matrix back into the DataContainer
                _insert_data_matrix(coupled_vis, data, ti, fi, pol)

def _apply_coupling_inverse(
    data,
    uvcoupling: UVCoupling,
    first_order: bool = False,
    multi_path: bool = False,
    inplace: bool = False
):
    """
    Apply coupling parameters to visibility data in reverse mode.
    """
    # Invert the coupling parameters if not already inverted
    if not uvcoupling.is_inverted:
        uvcoupling.invert(first_order, multi_path)

    for fi in range(uvcoupling.Nfreqs):
        for pi, pol in enumerate(uvcoupling.pols):
                for ti in range(uvcoupling.Ntimes):
                    
                    # Extract the visibility matrix for the current time, frequency, and polarization
                    vis_matrix = _extract_data_matrix(uvcoupling, data, ti, fi, pol)
        
                    # Apply the coupling parameters
                    if first_order:
                        # Apply first-order coupling
                        coupled_vis = np.dot(vis_matrix, uvcoupling.coupling[pi, :, :, ti, fi].T.conj())
                    else:
                        # Apply second-order or higher coupling
                        coupled_vis = np.dot(vis_matrix, uvcoupling.coupling[pi, :, :, ti, fi])

                    # Insert the modified visibility matrix back into the DataContainer
                    _insert_data_matrix(coupled_vis, data, ti, fi, pol)

def apply_coupling(
    data: DataContainer | np.ndarray,
    uvcoupling: UVCoupling,
    forward: bool = False,
    first_order: bool = False,
    multi_path: bool = False,
    inplace: bool = False,
) -> DataContainer:
    """
    TODO: Handle forward and reverse application of coupling parameters and 
          make it very clear which mode is being used.

    Apply coupling parameters to visibility data.

    Parameters
    ----------
    data : DataContainer or np.ndarray
        Visibility data to which the coupling parameters will be applied.
    uvcoupling : UVCoupling
        UVCoupling object containing coupling parameters.
    forward : bool, optional
        If True, apply coupling in forward mode (default is False, which applies in reverse mode).
    first_order : bool, optional
        If True, apply only first-order coupling terms. If False, use second-order terms.
    multi_path : bool, optional
        If True, apply multi-path coupling corrections.
    inplace : bool, optional
        If True, modify the input data in place. If False, return a new DataContainer.
        Default is False.

    Returns
    -------
    DataContainer
        Visibility data with coupling applied.
    """
    # Validate the input data against the coupling parameters
    uvcoupling._validate_data(data) # Probably should be a method outside of UVCoupling

    # Number of antennas
    nants = uvcoupling.nants  

    # Get identity matrix for the number of antennas
    identity = np.eye(nants, dtype=np.complex128)
    identity_shape = list(uvcoupling.shape)
    identity = np.broadcast_to(identity, identity_shape)

    if not inplace:
        # Create a deep copy of the data if not inplace
        data = copy.deepcopy(data)
    
    if forward:
        # Apply coupling in forward mode
        _apply_coupling_forward(data, uvcoupling, first_order, multi_path, inplace)
    else:
        # Apply coupling in reverse mode
        _apply_coupling_inverse(data, uvcoupling, first_order, multi_path, inplace)

    return data


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