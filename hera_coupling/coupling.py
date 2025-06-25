import h5py
import copy
import numpy as np
from scipy import linalg
from typing import Dict, Optional, Union, List, Tuple
from hera_cal.datacontainer import DataContainer
from pathlib import Path

from ._version import version

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
            pols: List[str],
            times: Optional[np.ndarray] = None,
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
        pols : List[str]
            Polarization strings (e.g., ['ee', 'nn']).
        times : np.ndarray, optional
            Time array in JD. Required if Ntimes > 1.
        """
        self.coupling = coupling
        self.antpos = dict(antpos)
        self.freqs = freqs
        self.times = times  
        self.pols = pols

        # Derived properties
        self.dtype = coupling.dtype
        self.ants = list(self.antpos.keys())
        (
            self.npols,
            self.nants,
            _,
            self.ntimes,
            self.nfreqs
        ) = self.coupling.shape

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

        # Create identity matrix for coupling
        self.identity_matrix = np.eye(self.nants, dtype=self.dtype)

    def _validate_shapes(self) -> None:
        """Validate that all arrays have consistent shapes."""
        if not self.production:
            expected_shape = (len(self.pols), self.nants, self.nants, 
                             self.ntimes, self.nfreqs)
            
            if self.coupling.shape != expected_shape:
                raise ValueError(f"coupling shape {self.coupling.shape} doesn't match "
                               f"expected {expected_shape}")
            
            if len(self.antpos) != self.nants:
                raise ValueError(f"antpos length {len(self.antpos)} != Nants {self.nants}")
                               
            if len(self.freqs) != self.nfreqs:
                raise ValueError(f"freqs length {len(self.freqs)} != Nfreqs {self.nfreqs}")
                
            if self.times is not None and len(self.times) != self.ntimes:
                raise ValueError(f"times length {len(self.times)} != ntimes {self.ntimes}")
            
            if self.times is None and self.ntimes > 1:
                raise ValueError("times must be provided if ntimes > 1")
        else:
            # In production mode, skip validation
            return
        
    def _validate_data(self, data: DataContainer | np.ndarray) -> None:
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
            if isinstance(data, DataContainer):
                if len(data.ants) != self.nants:
                    raise ValueError(f"Data has {len(data.antpos)} antennas, but coupling has {self.nants} antennas.")
                
                if len(data.freqs) != self.coupling.shape[-1]:
                    raise ValueError(f"Data has {len(data.freqs)} frequencies, but coupling has {self.coupling.shape[-1]} frequencies.")
                
                if not self.is_time_invariant and len(data.times) != self.coupling.shape[-2]:
                    raise ValueError(f"Data has {len(data.times)} times, but coupling has {self.coupling.shape[-2]} times.")
                
                for pol in self.pols:
                    if pol not in data.pols:
                        raise ValueError(f"Coupling polarization string {pol} not in data polarizations: {data.pols}.")
            elif isinstance(data, np.ndarray):
                # If data is a plain ndarray, check its shape
                if data.ndim != self.coupling.ndim:
                    raise ValueError(f"Data ndarray must have 5 dimensions, got {data.ndim}.")
                
                if data.shape[1] != self.nants or data.shape[2] != self.nants:
                    raise ValueError(f"Data shape {data.shape} does not match expected shape for coupling {self.coupling.shape}.")
                
                if (not self.is_time_invariant and data.shape[3] != self.ntimes) or data.shape[4] != self.nfreqs:
                    raise ValueError(f"Data shape {data.shape} does not match expected times and frequencies.")
                
                if data.shape[0] != len(self.pols):
                    raise ValueError(f"Data shape {data.shape} does not match number of polarizations {len(self.pols)}.")
            else:
                raise TypeError("data must be a DataContainer or a numpy ndarray")
        else:
            return  # In production mode, skip validation
            
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
        if self.is_inverted and self._inversion_cache['first_order'] == first_order and \
           self._inversion_cache['multi_path'] == multi_path:
            # If already inverted with the same parameters, return cached result
            return 
        
        if first_order:
            # Inverse not needed for first-order coupling solver
            self._inverted = False            
        else:
            # Initialize the inverse coupling matrix
            self.inverse_coupling = np.zeros_like(self.coupling, dtype=self.dtype)

            # Invert the coupling matrix for each polarization, time, and frequency
            for pi, pol in enumerate(self.pols):
                for ti in range(self.ntimes):
                    for fi in range(self.nfreqs):
                        # Extract the coupling matrix for the current polarization, time, and frequency
                        coupling_matrix = self.coupling[pi, :, :, ti, fi].copy()
            
                        if multi_path:
                            # If multi-path coupling is enabled, we need to dot the coupling matrix with itself
                            coupling_matrix += coupling_matrix.dot(coupling_matrix)

                        self.inverse_coupling[pi, :, :, ti, fi] = np.linalg.pinv(
                            coupling_matrix + self.identity_matrix
                        )

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

    def write_coupling(
        self,
        filename: str | Path, 
        clobber: bool=False,
        chunks: str=True,
        data_compression: str=None,
    ) -> None:
        """
        Function to write a coupling object out to a hdf5 file

        Parameters:
        ----------
        filename: str | Path
            Path to the output file where the coupling parameters will be saved.
        clobber: bool, optional
            If True, overwrite the file if it exists. Default is False.
        chunks: str, optional
            Chunking strategy for the dataset. Default is True, which uses automatic chunking.
        data_compression: str, optional
            Compression algorithm to use for the dataset. Default is None, which means no compression.

        Raises:
        ------
        FileExistsError
            If the file already exists and clobber is False.
        """
        filename = Path(filename)
            
        if filename.exists() and not clobber:
            raise FileExistsError(f"File {filename} exists. Use clobber=True to overwrite.")
        
        with h5py.File(filename, 'w') as f:
            # Create header for attribute info
            header = f.create_group("Header")
            
            # Write out version information
            header["version"] = np.bytes_(version)
            
            # Write out coordinate arrays
            header["freqs"] = self.freqs
            if self.times is not None:
                header["times"] = self.times
            else:
                header["times"] = h5py.Empty("f")

            # Extract antenna numbers (keys) and positions (values)
            ant_nums = list(self.antpos.keys())
            ant_positions = list(self.antpos.values())
            header["antenna_numbers"] = np.array(ant_nums, dtype=int)
            header["antenna_positions"] = np.array(ant_positions)

            # Store polarizations
            pols_encoded = [p.encode('utf-8') for p in self.pols]
            header["polarization_array"] = pols_encoded
            
            # Create data group for main arrays
            data_group = f.create_group("Data")
            data_group.create_dataset(
                'coupling',
                data=self.coupling,
                compression=data_compression,
                chunks=chunks
            )

    @classmethod
    def read_coupling(cls, filename: Union[str, Path]) -> 'UVCoupling':
        """
        Read a UVCoupling object from an HDF5 file.
        
        Parameters
        ----------
        filename : str or Path
            Path to the HDF5 file to read
            
        Returns
        -------
        UVCoupling
            The UVCoupling object loaded from the file
            
        Raises
        ------
        FileNotFoundError
            If the specified file does not exist
        ValueError
            If the file format is not recognized or corrupted
            
        Examples
        --------
        >>> uvc = read_coupling('coupling_data.h5')
        >>> print(f"Loaded coupling with shape: {uvc.coupling.shape}")
        """
        filename = Path(filename)
        
        if not filename.exists():
            raise FileNotFoundError(f"File {filename} does not exist")
        
        with h5py.File(filename, 'r') as f:
            # Check file format
            header = f['Header']
            data_group = f['Data']
            
            # Check version compatibility
            version = header['version'][()].decode('utf-8')
        
            # Read coordinate arrays
            freqs = header['freqs'][()]
            
            # Handle times (could be empty)
            times_data = header['times'][()]
            if isinstance(times_data, h5py.Empty):
                times = None
            else:
                times = times_data
            
            # Read antenna information
            antenna_numbers = header['antenna_numbers'][()]
            antenna_positions = header['antenna_positions'][()]
            
            # Reconstruct antpos dictionary
            antpos = dict(zip(antenna_numbers, antenna_positions))
            
            # Read polarizations
            pols_encoded = header['polarization_array'][()]
            pols = [p.decode('utf-8') for p in pols_encoded]
            
            # Read main coupling data
            coupling = data_group['coupling'][()]

        # Create and return UVCoupling object
        uvc = cls(
            coupling=coupling,
            freqs=freqs,
            times=times,
            antpos=antpos,
            pols=pols,
        )
        
        return uvc
    
def _extract_data_matrix(
    data: DataContainer, 
    antpos: Dict[int, np.ndarray],
    time_idx: int, 
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
    vis_matrix = np.zeros((data.freqs.size, nants, nants), dtype=complex)
    
    # Fill matrix from data
    for ant1 in antpos:
        for ant2 in antpos:
            if ant1 > ant2:
                continue

            blpol = (ant1, ant2, pol)
            i, j = ant_to_idx[ant1], ant_to_idx[ant2]
            
            # Get visibility data
            vis_matrix[..., i, j] = data[blpol][time_idx]

            if i != j:  # Add conjugate for off-diagonal
                vis_matrix[..., j, i] = np.conj(data[blpol][time_idx])
                
    return vis_matrix

def _insert_data_matrix(
    vis_matrix: np.ndarray, 
    data: DataContainer, 
    antpos: Dict[int, np.ndarray],
    time_idx: int,  
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

    for ant1 in antpos:
        for ant2 in antpos:
            if ant1 > ant2:
                continue
            blpol = (ant1, ant2, pol)
            i, j = ant_to_idx[ant1], ant_to_idx[ant2]
                
            # Store visibility data
            data[blpol][time_idx] = vis_matrix[..., i, j]

            if i != j:  # Store conjugate for off-diagonal
                data[(ant2, ant1, pol)][time_idx] = vis_matrix[..., j, i]


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
    identity = uvcoupling.identity_matrix

    if isinstance(data, DataContainer):
        ntimes = len(data.times)
        nfreqs = len(data.freqs)
    elif isinstance(data, np.ndarray):
        # If data is a plain ndarray, assume it has shape (Npols, Nants, Nants, Ntimes, Nfreqs)
        ntimes = data.shape[3]
        nfreqs = data.shape[4]
    else:
        raise TypeError("data must be a DataContainer or a numpy ndarray")
    
    for pi, pol in enumerate(uvcoupling.pols):
        for ti in range(ntimes):
            if isinstance(data, DataContainer):
                vis_matrix = _extract_data_matrix(
                    data=data, 
                    antpos=uvcoupling.antpos, 
                    time_idx=ti, 
                    pol=pol
                )
            else:
                # If data is a plain ndarray, assume it has shape (Npols, Nants, Nants, Ntimes, Nfreqs)
                vis_matrix = np.transpose(
                    data[pi, :, :, ti, :], (2, 0, 1)
                )
            coupled_vis = np.zeros_like(vis_matrix)
            for fi in range(nfreqs):
                # Extract the visibility matrix for the current time, frequency, and polarization
                if uvcoupling.is_time_invariant:
                    # If time-invariant, use the first frequency index
                    coupling_matrix = uvcoupling.coupling[pi, :, :, 0, fi].copy()
                else:
                    coupling_matrix = uvcoupling.coupling[pi, :, :, ti, fi].copy()

                if multi_path and not first_order:
                    # If multi-path coupling is enabled, we need to dot the coupling matrix with itself
                    coupling_matrix += coupling_matrix.dot(coupling_matrix)

                # Apply the coupling parameters
                if first_order:
                    # Apply first-order coupling
                    _coupled_vis = (
                        vis_matrix[fi] +
                        np.dot(vis_matrix[fi], coupling_matrix.T.conj()) + 
                        np.dot(vis_matrix[fi], coupling_matrix.T.conj()).T.conj()
                    )
                    
                else:
                    # Apply second-order or higher coupling
                    coupling_matrix += identity
                    _coupled_vis = np.dot(
                        np.dot(coupling_matrix, vis_matrix[fi]),
                        coupling_matrix.T.conj()
                    )
                
                # Store the coupled visibility matrix
                coupled_vis[fi] = _coupled_vis

            if isinstance(data, DataContainer):
                # Insert the modified visibility matrix back into the DataContainer
                _insert_data_matrix(
                    vis_matrix=coupled_vis, 
                    data=data, 
                    antpos=uvcoupling.antpos,
                    time_idx=ti,
                    pol=pol
                )
            else:
                # If data is a plain ndarray, modify it in place
                data[pi, :, :, ti, :] = np.transpose(
                    coupled_vis, (1, 2, 0)
                )

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
    
    # Build an identity matrix for the number of antennas
    identity = uvcoupling.identity_matrix

    if isinstance(data, DataContainer):
        # If data is a DataContainer, get the number of times and frequencies
        ntimes = len(data.times)
        nfreqs = len(data.freqs)
    elif isinstance(data, np.ndarray):
        # If data is a plain ndarray, assume it has shape (Npols, Nants, Nants, Ntimes, Nfreqs)
        ntimes = data.shape[3]
        nfreqs = data.shape[4]
    else:
        raise TypeError("data must be a DataContainer or a numpy ndarray")
    
    for pi, pol in enumerate(uvcoupling.pols):
        for ti in range(ntimes):
            if isinstance(data, DataContainer):
                # Extract the visibility matrix for the current time and polarization
                vis_matrix = _extract_data_matrix(
                    data=data, 
                    antpos=uvcoupling.antpos, 
                    time_idx=ti, 
                    pol=pol
                )
            else:
                # If data is a plain ndarray, assume it has shape (Npols, Nants, Nants, Ntimes, Nfreqs)
                vis_matrix = np.transpose(
                    data[pi, :, :, ti, :], (2, 0, 1)
                )

            uncoupled_vis = np.zeros_like(vis_matrix)
            for fi in range(nfreqs):
                # Apply the coupling parameters
                if first_order:
                    if uvcoupling.is_time_invariant:
                        # If time-invariant, use the first frequency index
                        coupling_matrix = uvcoupling.coupling[pi, :, :, 0, fi]
                    else:
                        coupling_matrix = uvcoupling.coupling[pi, :, :, ti, fi]

                    # Apply first-order coupling, Sylvester solver
                    _uncoupled_vis = linalg.solve_sylvester(
                        coupling_matrix, identity + coupling_matrix.T.conj(), vis_matrix[fi]
                    )
                else:
                    # Get inverse coupling matrix
                    if uvcoupling.is_time_invariant:
                        # If time-invariant, use the first frequency index
                        inverse_coupling = uvcoupling.inverse_coupling[pi, :, :, 0, fi]
                    else:
                        inverse_coupling = uvcoupling.inverse_coupling[pi, :, :, ti, fi]

                    # Apply second-order or higher coupling
                    _uncoupled_vis = np.dot(
                        np.dot(inverse_coupling, vis_matrix[fi]),
                        inverse_coupling.T.conj()
                    )

                # Store the uncoupled visibility matrix
                uncoupled_vis[fi] = _uncoupled_vis

            if isinstance(data, DataContainer):
                # Insert the modified visibility matrix back into the DataContainer
                _insert_data_matrix(
                    vis_matrix=uncoupled_vis, 
                    data=data, 
                    antpos=uvcoupling.antpos,
                    time_idx=ti,  
                    pol=pol
                )
            else:
                # If data is a plain ndarray, modify it in place
                data[pi, :, :, ti, :] = np.transpose(
                    uncoupled_vis, (1, 2, 0)
                )

def apply_coupling(
    data: DataContainer | np.ndarray,
    uvcoupling: UVCoupling,
    forward: bool = False,
    first_order: bool = False,
    multi_path: bool = False,
    inplace: bool = False,
) -> DataContainer:
    """
    Apply coupling parameters to visibility data. This function can apply coupling in both forward and 
    inverse modes, depending on the `forward` parameter.

    Parameters
    ----------
    data : DataContainer or np.ndarray
        Visibility data to which the coupling parameters will be applied.
    uvcoupling : UVCoupling
        UVCoupling object containing coupling parameters.
    forward : bool, optional
        If True, apply coupling in forward mode (default is False, which applies in inverse mode).
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
    uvcoupling._validate_data(data)

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