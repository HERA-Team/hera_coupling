import h5py
import copy
import numpy as np
from scipy import linalg
from typing import Dict, Optional, Union, List, Tuple
from hera_cal.datacontainer import DataContainer

from .solvers import SylvesterSolver

class UVCoupling:
    """
    A data format for antenna-to-antenna mutual coupling parameters.
    
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
        self._is_time_invariant = (coupling.shape[-2] == 1) and self.times is None  # Default to time-invariant
        self._inverted = False # Inversion status
        self._inversion_cache = {}  # Cache for inverted coupling if needed

        # Validate shapes
        self.set_production(False)
        self._validate_shapes()
        
        # Create antenna index mapping
        self.ant_to_idx = {ant: i for i, ant in enumerate(self.ants)}
        self.idx_to_ant = {i: ant for ant, i in self.ant_to_idx.items()}

    def _validate_shapes(self) -> None:
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
        
    def _validate_data(self, data: DataContainer) -> None:
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
            if len(data.ants) != self.Nants:
                raise ValueError(f"Data has {data.Nants} antennas, but coupling has {self.Nants} antennas.")
            
            if len(data.freqs) != self.coupling.shape[-1]:
                raise ValueError(f"Data has {data.Nfreqs} frequencies, but coupling has {self.coupling.shape[-1]} frequencies.")
            
            if not self.is_time_invariant and len(data.times) != self.coupling.shape[-2]:
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
        Invert the coupling matrix for use in applying inverse coupling to visibility data.

        Parameters
        ----------
        first_order : bool
            If True, use first-order approximation for coupling. If this is True, the coupling matrix is not inverted
            because the first-order approximation does not require an inverse.
        multi_path : bool
            If True, include multi-path terms in the inversion. This means that the coupling matrix will be modified to 
            include multi-path effects.
        """
        # TODO:
        # 1. We may want to invert this once and store it, as such
        #    we may want to have self.inverted = True/False
        #  
        # TODO: we should probably cache based on the assumption of first order / multi-path
        if self.is_inverted and self._inversion_cache['first_order'] == first_order and \
           self._inversion_cache['multi_path'] == multi_path:
            # If already inverted with the same parameters, return cached result
            return 
        
        if self.is_inverted and first_order:
            # Inverse not needed for first-order coupling solver
            self._inverted = False            
        elif not first_order:
            # Initialize the inverse coupling matrix
            self.inverse_coupling = np.zeros_like(self.coupling, dtype=np.complex128)
            identity = np.eye(self.Nants, dtype=np.complex128)

            # Invert the coupling matrix for each polarization, time, and frequency
            for pi, pol in enumerate(self.pols):
                for ti in range(self.Ntimes):
                    for fi in range(self.Nfreqs):
                        # Extract the coupling matrix for the current polarization, time, and frequency
                        coupling_matrix = self.coupling[pi, :, :, ti, fi]
                        
                        if multi_path:
                            # If multi-path coupling is enabled, we need to dot the coupling matrix with itself
                            coupling_matrix += coupling_matrix.dot(coupling_matrix)

                        self.inverse_coupling[pi, :, :, ti, fi] = np.linalg.pinv(coupling_matrix + identity)

            self._inversion_cache['first_order'] = first_order
            self._inversion_cache['multi_path'] = multi_path
            self._inverted = True

    def apply(
            self, 
            data: DataContainer, 
            forward: bool = True,
            first_order: bool = False,
            multi_path: bool = False,
            inplace: bool = False
        ):
        """
        Apply coupling to visibility data.
        
        Parameters  
        ----------
        data : DataContainer or np.ndarray
            Visibility data
        forward : bool, optional
            If True, apply forward coupling. If False, apply inverse coupling.
        first_order : bool, optional
            Use first-order approximation.
        multi_path : bool, optional
            Include multi-path terms.
        inplace : bool, optional
            Modify data in place.
            
        Returns
        -------
        DataContainer or np.ndarray
            Coupled visibility data
            
        Examples
        --------
        >>> # Apply forward coupling
        >>> coupled_data = uvc.apply(data, forward=True)
        
        >>> # Apply inverse coupling to remove coupling effects  
        >>> uncoupled_data = uvc.apply(coupled_data, forward=False)
        """
        return apply_coupling(
            data=data,
            uvcoupling=self,
            forward=forward,
            first_order=first_order,
            multi_path=multi_path,
            inplace=inplace
        )

    def write_coupling(self, filename: str, **kwargs) -> None:
        """
        TODO: Write the coupling parameters to an hdf5 file.

        What are the necessary key-value pairs to store?
            - coupling: ndarray
            - Antenna names: ndarray
            - Antenna positions: dict
            - Frequencies: ndarray
            - Times: ndarray (if not time-invariant)
        """
        raise NotImplementedError

    @classmethod
    def read_coupling(cls, filename: str, **kwargs) -> "UVCoupling":
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
        data = read_uvcoupling(filename)

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
    
def _validate_data(data: DataContainer, uvc: UVCoupling) -> None:
    """
    Validate that the input data is compatible with the UVCoupling parameters.
    
    Parameters
    ----------
    data : DataContainer
        The visibility data to validate.
    uvc : UVCoupling
        The UVCoupling object containing coupling parameters.
    
    Raises
    ------
    ValueError
        If the data does not match the expected shapes or dimensions.
    """
    uvc._validate_data(data)  # Call the validation method in UVCoupling
    
def _extract_data_matrix(
    data: DataContainer, 
    antpos: Dict[int, np.ndarray],
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
    # Create antenna index mapping
    ant_to_idx = {ant: i for i, ant in enumerate(antpos)}
    nants = len(ant_to_idx)
    
    # Initialize visibility matrix
    vis_matrix = np.zeros((nants, nants), dtype=complex)
    
    # Fill matrix from data
    for bl in data.antpairs:
        if pol in data[bl]:
            ant1, ant2 = bl
            i, j = ant_to_idx[ant1], ant_to_idx[ant2]
            
            # Get visibility data
            vis_data = data[bl][pol]
            vis_value = vis_data[time_idx, freq_idx] 
            vis_matrix[i, j] = vis_value

            if i != j:  # Add conjugate for off-diagonal
                vis_matrix[j, i] = np.conj(vis_value)
                
    return vis_matrix

def _insert_data_matrix(
    vis_matrix: np.ndarray, 
    data: DataContainer, 
    antpos: Dict[int, np.ndarray],
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
    # Create antenna index mapping
    ant_to_idx = {ant: i for i, ant in enumerate(antpos)}

    for bl in data.antpairs:
        if pol in data[bl]:
            ant1, ant2 = bl
            i, j = ant_to_idx[ant1], ant_to_idx[ant2]
            
            # Store visibility data
            data[bl][pol][time_idx, freq_idx] = vis_matrix[i, j]


def _apply_coupling_forward(
    data: DataContainer | np.ndarray,
    uvcoupling: UVCoupling,
    first_order: bool = False,
    multi_path: bool = False,
):
    """
    Apply coupling parameters to visibility data in forward mode.

    Parameters
    ----------
    data : DataContainer or np.ndarray
        Visibility data to which the coupling parameters will be applied.
    uvcoupling : UVCoupling
        UVCoupling object containing coupling parameters.
    first_order : bool, optional
        If True, apply only first-order coupling terms. If False, use second-order terms.
    multi_path : bool, optional
        If True, apply multi-path coupling corrections.
    """
    # Build an identity matrix for the number of antennas
    nants = uvcoupling.Nants
    identity = np.eye(nants, dtype=np.complex128)

    for ti in range(uvcoupling.Ntimes):
        for fi in range(uvcoupling.Nfreqs):
            for pi, pol in enumerate(uvcoupling.pols):
                # Extract the visibility matrix for the current time, frequency, and polarization
                vis_matrix = _extract_data_matrix(data, uvcoupling.antpos, ti, fi, pol)
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
    data: DataContainer | np.ndarray,
    uvcoupling: UVCoupling,
    first_order: bool = False,
    multi_path: bool = False,
) -> None:
    """
    Apply coupling parameters to visibility data in reverse mode.

    Parameters
    ----------
    data : DataContainer or np.ndarray
        Visibility data to which the coupling parameters will be applied.
    uvcoupling : UVCoupling
        UVCoupling object containing coupling parameters.
    first_order : bool, optional
        If True, apply only first-order coupling terms. If False, use second-order terms.
    multi_path : bool, optional
        If True, apply multi-path coupling corrections.
    inplace : bool, optional
        If True, modify the input data in place. If False, return a new DataContainer.
        Default is False.
    """
    # Invert the coupling parameters if not already inverted
    uvcoupling.invert(first_order=first_order, multi_path=multi_path)
    identity = np.squeeze(uvcoupling.identity_matrix)

    for fi in range(uvcoupling.Nfreqs):
        for pi, pol in enumerate(uvcoupling.pols):
            if first_order and uvcoupling.is_time_invariant:
                solver = SylvesterSolver(
                    uvcoupling.coupling[pi, :, :, 0, fi],
                    identity + uvcoupling.coupling[pi, :, :, 0, fi].T.conj(),
                )
            for ti in range(uvcoupling.Ntimes):
                # Extract the visibility matrix for the current time, frequency, and polarization
                vis_matrix = _extract_data_matrix(data, uvcoupling.antpos, ti, fi, pol)
    
                # Apply the coupling parameters
                if first_order:
                    # Apply first-order coupling, Sylvester solver
                    if uvcoupling.is_time_invariant:
                        uncoupled_vis = solver.solve(vis_matrix)
                    else:
                        uncoupled_vis = linalg.solve_sylvester(
                            uvcoupling.coupling[pi, :, :, ti, fi],
                            identity + uvcoupling.coupling[pi, :, :, ti, fi].T.conj(),
                            vis_matrix
                        )
                else:
                    # Get inverse coupling matrix
                    inverse_coupling = uvcoupling.inverse_coupling[pi, :, :, ti, fi]

                    # Apply second-order or higher coupling
                    uncoupled_vis = np.dot(
                        np.dot(inverse_coupling, uvcoupling.coupling[pi, :, :, ti, fi]),
                        inverse_coupling.T.conj()
                    )

                # Insert the modified visibility matrix back into the DataContainer
                _insert_data_matrix(uncoupled_vis, data, ti, fi, pol)

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
    _validate_data(data, uvcoupling)

    if not inplace:
        # Create a deep copy of the data if not inplace
        data = copy.deepcopy(data)
    
    if forward:
        # Apply coupling in forward mode
        _apply_coupling_forward(
            data=data, 
            uvcoupling=uvcoupling, 
            first_order=first_order, 
            multi_path=multi_path
        )
    else:
        # Apply coupling in reverse mode
        _apply_coupling_inverse(
            data=data, 
            uvcoupling=uvcoupling, 
            first_order=first_order, 
            multi_path=multi_path, 
        )

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