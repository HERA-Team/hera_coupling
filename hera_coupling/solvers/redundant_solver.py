import copy
import numpy as np
from functools import partial
from astropy.constants import c
from scipy.fft import next_fast_len
from typing import Tuple, Union, List, Dict, Optional

# Jax libraries
import jax
import tqdm
import jaxopt
import optax
from jax import numpy as jnp
# jax.config.update("jax_enable_x64", True)

from hera_filters import dspec
from hera_cal import utils
from hera_cal.redcal import get_pos_reds
from hera_cal.datacontainer import DataContainer

from hera_coupling import UVMutualCoupling
"""
Outstanding Issues:
1. The `project_baselines_to_grid` function needs to be integrated with the coupling grid
   to ensure that the antenna positions are correctly projected onto the grid.
2. The `RedundantCouplingManager` class needs to be fully implemented to handle the coupling parameters
   and provide methods for selecting and setting coupling values.
3. Need to decide if the 

"""


class RedundantCouplingManager:
    """
    Class that handles the coupling grid for antenna positions. This class assumes that the 
    antenna positions are able to be defined on a 2D grid.
    """
    def __init__(self, active_antpos, tol: float = 1e-2, ew_pair=(0, 1), ns_pair=(0, 11), ratio: int = 3, pad_grid: bool = True):
        """
        Initialize the RedundantCouplingManager with the active antenna positions and prepare the coupling grid.

        Parameters:
        -----------
            active_antpos : dictionary
                Antenna positions of the active antennas in the array
            include_autos : bool, default=False
                Flag on whether to include auto correlations in visibility grid
            tol : float, default=1e-2
                Tolerance for rounding the coordinates to the nearest integer grid.
            ew_pair : tuple of int, default=(0, 1)
                Antenna indices (i, j) that define the reference east-west direction.
            ns_pair : tuple of int, default=(0, 11)
                Antenna indices (i, j) that define the reference north-south direction.
            ratio : int, default=3
                Scaling factor to normalize unit vectors (default is 3).
            pad_grid : bool, default=True
                Whether to pad the grid shape to at least twice the maximum coordinate value.
        """
        # Store the active antenna positions and prepare the redundant groups
        self.active_antpos = active_antpos
        self.all_reds = get_pos_reds(            
            active_antpos, 
            include_autos=True,
        )
        self.tol = tol

        # Extract a representative list of antenna pairs for each redundant group
        self.antpairs = [red[0] for red in self.all_reds]

        # Initialize grid properties
        self.bl_to_grid_coords = {}
        self.grid_coord_to_bl = {}
        self.grid_shape = None

        # Prepare the coordinates for the coupling grid
        # This will project the antenna pairs onto a 2D grid based on their positions
        self.prepare_coordinates(
            antpairs=self.antpairs, 
            antpos=self.active_antpos,
            tol=tol,
            ew_pair=ew_pair,  # Default east-west pair
            ns_pair=ns_pair,  # Default north-south pair
            ratio=ratio,  # Default ratio for scaling the unit vectors
            pad_grid=pad_grid,  # Default padding for the grid shape
        )

    def prepare_coordinates(self, antpairs, antpos, tol: float=1e-2, ew_pair=(0, 1), ns_pair=(0, 11), ratio: int = 3, pad_grid: bool = True):
        """
        Prepare the coordinates for the coupling grid based on the antenna pairs and positions.

        Parameters:
        -----------
            antpairs : list of tuple of int
                List of antenna index pairs [(i1, j1), (i2, j2), ...] to project.
            antpos : array_like
                Array of antenna positions with shape (N_antennas, 3).

        Returns:
        --------
            np.ndarray
                An array of shape (len(antpairs), 2) with [east, north] projections 
                for each baseline.
        """
        # Project the antenna pairs onto a 2D grid
        coordinates = project_coordinates_to_grid(
            antpairs, 
            antpos,
            ew_pair=ew_pair,  # Default east-west pair
            ns_pair=ns_pair,  # Default north-south pair
            ratio=ratio  # Default ratio for scaling the unit vectors
        )

        # Round the coordinates to the nearest integer and convert to int
        gridded_coords = np.round(coordinates, decimals=0).astype(int)

        # Ensure that the coordinates are within the specified tolerance of the integer grid
        # This checks that the absolute difference between the original and gridded coordinates
        assert np.less_equal(np.abs(coordinates - gridded_coords), tol).all(), \
            "Coordinates not within tolerance of integer grid."
        
        # Set the grid shape based on the maximum coordinates
        max_coords = np.max(np.max(gridded_coords, axis=0) - np.min(gridded_coords, axis=0))
        if pad_grid:
            self.grid_shape = next_fast_len(int(2 * max_coords) + 1)
        else:
            self.grid_shape = next_fast_len(int(1.1 * max_coords) + 1)
        
        # Create a mapping from antenna pairs to their gridded coordinates
        self.bl_to_grid_coords = {
            antpair: coord # Use the gridded coordinates as the value
            for antpair, coord in zip(antpairs, gridded_coords)
        }
        self.bl_to_grid_coords.update({
            (antpair[1], antpair[0]): -coord # Use the gridded coordinates as the value
            for antpair, coord in zip(antpairs, gridded_coords)
        })

        # Create a mapping from gridded coordinates to antenna pairs
        self.grid_coord_to_bl = {
            tuple(coord): bl for bl, coord in self.bl_to_grid_coords.items()
        }

def build_data_and_coupling_grids(
    coupling_manager: RedundantCouplingManager,
    data: DataContainer,
    flags: DataContainer,
    nsamples: DataContainer,
    *,
    pol: str='ee',
    time_slice=slice(0, None),
    freq_slice=slice(0, None),
    window_function: str='hann',
    skip_autos: bool=False,
    compressed=False,
    axes=(0, 1),
    **kwargs,
):
    """
    TODO: should build a function that finds a good range to pull from

    Build the data grid and coupling arrays for the given data and coupling manager.

    Parameters:
    -----------
    coupling_manager : RedundantCouplingManager
        The coupling manager that handles the antenna positions and grid coordinates.
    data : DataContainer
        The data container containing the visibilities to be gridded.
    flags : DataContainer
        The flags for the data, indicating which visibilities are valid.
    nsamples : DataContainer
        The number of samples for each visibility in the data container.
    pol : str, optional
        Polarization string (e.g., 'ee', 'nn', etc.) to select the appropriate data (default: 'ee').
    time_slice : slice, optional
        Slice for the time dimension of the data (default: slice(0, None)).
    freq_slice : slice, optional
        Slice for the frequency dimension of the data (default: slice(0, None)).
    window_function : str, optional
        The window function to apply to the data (default: 'hann').
    **kwargs : dict
        Additional keyword arguments for filtering baselines, such as `max_len`, `max_ew`, `max_ns`, etc.

    Returns:
    --------
    data_grid : np.ndarray
        The gridded data array of shape (ntimes, nfreqs, ngrid, ngrid).
    noise_array : np.ndarray
        The noise variance array for the coupling parameters of shape (1, nfreqs, ncoupling_antpairs).
    coupling_array : np.ndarray
        The coupling parameters array of shape (1, nfreqs, ncoupling_antpairs).
    coupling_idx : np.ndarray
        The indices of the coupling parameters in the grid of shape (ncoupling_antpairs, 2).
    """
    # Start by producing the noise variance for the given time and frequency slices
    noise_var, window = estimate_windowed_noise_variance(
        data=data,
        flags=flags,
        nsamples=nsamples,
        time_slice=time_slice,
        freq_slice=freq_slice,
        window_function=window_function,
        axes=axes,
    )

    # Get all of the data which are unflagged over all times and frequencies
    unflagged_baselines = []
    all_baselines = [] # TODO: Temporary change to fix ordering error
    for key in data:
        if pol in key:
            all_baselines.append(key[:2])
            if np.all(~flags[key][time_slice][:, freq_slice]):
                unflagged_baselines.append(key[:2])

    # Get the usable baselines of the baselines that have been found to be unflagged
    # over the time and frequency range
    usable_baselines = filter_baselines(
        all_baselines,
        antpos=coupling_manager.active_antpos,
        bls=unflagged_baselines, # Use all unflagged baselines
        **kwargs # Pass along arguments for filtering
    )
    
    # Reformat the data from DataContainer to jax array
    data_grid, noise_array, data_idx, coupling_array, coupling_idx, fit_idx = _build_data_and_coupling_arrays(
        coupling_manager=coupling_manager,
        data=data,
        noise_var=noise_var,
        data_antpairs=unflagged_baselines,
        coupling_antpairs=usable_baselines, # TODO: Not a great name for a parameter
        pol=pol,
        time_slice=time_slice,
        freq_slice=freq_slice,
        skip_autos=skip_autos,
        compressed=compressed,
    )

    return data_grid, noise_array, data_idx, coupling_array, coupling_idx, fit_idx, window

def _build_data_and_coupling_arrays(
    coupling_manager: RedundantCouplingManager,
    data: DataContainer,
    noise_var: DataContainer,
    data_antpairs: List[Tuple[int, int]],
    coupling_antpairs: List[Tuple[int, int]],
    pol: str,
    *,
    time_slice: slice=slice(0, None),
    freq_slice: slice=slice(0, None),
    skip_autos: bool=False,
    compressed: bool=False,
):
    """
    Build the data grid and coupling arrays for the given data and coupling manager.

    Parameters:
    -----------
    coupling_manager : RedundantCouplingManager
        The coupling manager that handles the antenna positions and grid coordinates.
    data : DataContainer
        The data container containing the visibilities to be gridded.
    noise_var : DataContainer
        The noise variance for the data, used to scale the coupling parameters.
    data_antpairs : list of tuple of int
        List of antenna pairs for the data to be gridded.
    coupling_antpairs : list of tuple of int
        List of antenna pairs for the coupling parameters.
    pol : str
        Polarization string (e.g., 'ee', 'nn', etc.) to select the appropriate data.
    time_slice : slice, optional
        Slice for the time dimension of the data (default: slice(0, None)).
    freq_slice : slice, optional
        Slice for the frequency dimension of the data (default: slice(0, None)).
    compressed : bool, optional
        Whether to compress the data grid (default: False).

    Returns:
    --------
    data_grid : np.ndarray
        The gridded data array of shape (ntimes, nfreqs, ngrid, ngrid).
    noise_array : np.ndarray
        The noise variance array for the coupling parameters of shape (1, nfreqs, ndata_antpairs).
    data_idx : np.ndarray
        The indices of the data in the grid of shape (ndata_antpairs, 2).
    coupling_array : np.ndarray
        The coupling parameters array of shape (1, nfreqs, ncoupling_antpairs).
    coupling_idx : np.ndarray
        The indices of the coupling parameters in the grid of shape (ncoupling_antpairs, 2).
    """
    # Get the data shape for the grid size
    ntimes, nfreqs = data[data_antpairs[0] + (pol,)][time_slice][:, freq_slice].shape

    grid_coords = np.array([coupling_manager.bl_to_grid_coords[bl] for bl in data_antpairs])
    diff = np.abs(np.max(grid_coords, axis=0) - np.min(grid_coords, axis=0)).max()
    ngrid = next_fast_len(int(1.1 * diff) + 1)
    
    # Build a grid for the data
    if compressed:
        data_array = []
    else:
        data_array = np.zeros((ntimes, nfreqs, ngrid, ngrid), dtype=complex)

    coupling_array = np.zeros((1, nfreqs, len(coupling_antpairs)), dtype=complex)
    
    # Initialize arrays for noise variance and indices
    noise_array = []
    data_idx = []
    fit_idx = []
    
    for ci, (ai, aj) in enumerate(data_antpairs):
        if ai == aj:
            if skip_autos:
                continue
            i, j = 0, 0  # Autos centered at the origin by default
        else:
            # Get the grid coordinates for the baseline
            i, j = coupling_manager.bl_to_grid_coords[(ai, aj)]

            # Only optimize the coupling parameters for non-auto baselines
            fit_idx.append((i, j))
            noise_array.append(noise_var[(ai, aj, pol)]) 

        # Add indices to the data_idx list
        data_idx.append((i, j)) 

        # Load data into grid or compressed array
        if compressed:
            data_array.append(data[(ai, aj, pol)][time_slice][:, freq_slice])
        else:
            data_array[:, :, i, j] = data[(ai, aj, pol)][time_slice][:, freq_slice]
            data_array[:, :, -i, -j] = data[(aj, ai, pol)][time_slice][:, freq_slice]

    if compressed:
        # Transpose to (ntimes, nfreqs, nbls)
        data_array = np.transpose(data_array, (1, 2, 0))

    # Array for mapping coupling parameters to grid positions
    coupling_idx = np.zeros((len(coupling_antpairs), 2), dtype=int)
    
    for ci, (ai, aj) in enumerate(coupling_antpairs):
        blmag = np.linalg.norm(data.antpos[aj] - data.antpos[ai])
        u = data.freqs[freq_slice] * blmag / c.value
        coupling_array[..., ci] = 1e-2 * (np.exp(2j * np.pi * u) / u)[None]
        coupling_idx[ci] = coupling_manager.bl_to_grid_coords[(ai, aj)]

    # Transpose array to shape (1, 1, nbls)
    noise_array = np.transpose(noise_array, (1, 2, 0))

    # Convert lists to numpy arrays
    data_idx = np.array(data_idx, dtype=int)
    fit_idx = np.array(fit_idx, dtype=int)
    coupling_idx = np.array(coupling_idx, dtype=int)

    return data_array, noise_array, data_idx, coupling_array, coupling_idx, fit_idx

def filter_baselines(
    all_bls: List[Tuple[int, int]],
    *,
    antpos: Dict[int, np.ndarray] = None,
    bls: Optional[List[Tuple[int, int]]] = None,
    ex_bls: Optional[List[Tuple[int, int]]] = None,
    max_len: Optional[float] = None,
    max_ew: Optional[float] = None,
    max_ns: Optional[float] = None,
) -> List[Tuple[int, int]]:
    """
    Filter baselines by inclusion/exclusion lists and maximum lengths while preserving order.

    Parameters
    ----------
    all_bls : list of tuple of int
        List of all baseline pairs [(i1, j1), (i2, j2), ...].
    antpos : dict, optional
        Dictionary of antenna positions {ant_index: position_vector}.
    bls : list of tuple of int, optional
        List of baselines to include. If provided, only these baselines will be kept.
    ex_bls : list of tuple of int, optional
        List of baselines to exclude. If provided, these baselines will be removed.
    max_len : float, optional
        Maximum length of the baseline to keep. If provided, only baselines with length <= max_len will be kept.
    max_ew : float, optional
        Maximum east-west component of the baseline to keep. If provided, only baselines with ew <= max_ew will be kept.
    max_ns : float, optional
        Maximum north-south component of the baseline to keep. If provided, only baselines with ns <= max_ns will be kept.

    Returns
    -------
    list of tuple of int
        Filtered list of baseline pairs that meet the specified criteria, preserving original order.
    """
    if bls is not None and ex_bls is not None:
        raise ValueError("Only one of `bls` or `ex_bls` may be provided.")
    
    # Create sets for O(1) lookup while preserving order through iteration
    if bls is not None:
        keep_set = set(bls)
    if ex_bls is not None:
        drop_set = set(ex_bls)
    
    # Check if we need length calculations
    need_length_calc = any(x is not None for x in [max_len, max_ew, max_ns])
    if need_length_calc:
        assert antpos is not None, "antpos must be provided if filtering baselines by length"
    
    filtered = []
    
    # Single pass through all_bls to preserve order
    for bl in all_bls:
        ant1, ant2 = bl
        
        # Skip auto-baselines
        if ant1 == ant2:
            continue
            
        # Apply inclusion/exclusion filters
        if bls is not None and bl not in keep_set:
            continue
        if ex_bls is not None and bl in drop_set:
            continue
            
        # Apply length-based filters if needed
        if need_length_calc:
            vec = antpos[ant2] - antpos[ant1]
            total = np.linalg.norm(vec)
            ew = abs(vec[0])
            ns = abs(vec[1])
            
            if max_len is not None and total > max_len:
                continue
            if max_ew is not None and ew > max_ew:
                continue
            if max_ns is not None and ns > max_ns:
                continue
        
        # Baseline passed all filters
        filtered.append(bl)
    
    return filtered

def project_coordinates_to_grid(antpairs, antpos, ew_pair=(0, 1), ns_pair=(0, 11), ratio: int = 3):
    """
    Projects baseline vectors between antenna pairs onto a 2D coordinate system 
    defined by approximate east-west and north-south directions.

    Parameters
    ----------
    antpairs : list of tuple of int
        List of antenna index pairs [(i1, j1), (i2, j2), ...] to project.
    antpos : array_like
        Array of antenna positions with shape (N_antennas, 3).
    ew_pair : tuple of int, optional
        Antenna indices (i, j) that define the reference east-west direction.
        Default is (0, 1).
    ns_pair : tuple of int, optional
        Antenna indices (i, j) that define the reference north-south direction.
        Default is (0, 11).
    ratio : int, optional
        Scaling factor to normalize unit vectors (default is 3).

    Returns
    -------
    np.ndarray
        An array of shape (len(antpairs), 2) with [east, north] projections 
        for each baseline.
    """
    # Define scaled reference vectors
    unit_ew = (antpos[ew_pair[1]] - antpos[ew_pair[0]]) / ratio
    unit_ns = (antpos[ns_pair[1]] - antpos[ns_pair[0]]) / ratio

    # Orthogonalize NS with respect to EW
    unit_vec_ns = unit_ns - np.dot(unit_ns, unit_ew) / np.linalg.norm(unit_ew) ** 2 * unit_ew

    projections = []
    for ap1, ap2 in antpairs:
        vec = antpos[ap2] - antpos[ap1]
        north = np.dot(vec, unit_vec_ns) / np.linalg.norm(unit_vec_ns) ** 2
        east = np.dot(vec - north * unit_ns, unit_ew) / np.linalg.norm(unit_ew) ** 2
        projections.append([east, north])

    return np.array(projections)


@jax.jit
def _scaled_log_1p_normalized(data):
    """
    Computes the scaled log(x + 1) function.

    Parameters
    ----------
        data : jnp.array
            The input data.
        alpha : float
            The scaling factor.
    
    Returns
    -------
        jnp.array
            The scaled log(1 + x) values.
    """
    return jnp.log1p(data)

@partial(jax.jit, static_argnames=['ngrid'])
def grid_coupling_array(data, indices, ngrid):
    """
    Grid baseline data onto a 2D grid using indices.
    
    Args:
        data: Array of shape (..., nbls) containing baseline data
        indices: Array of shape (nbls, 2) containing grid positions for each baseline
        ngrid: Size of the square grid
    
    Returns:
        Gridded array of shape (..., ngrid, ngrid)
    """
    batch_shape = data.shape[:-1]
    grid = jnp.zeros(batch_shape + (ngrid, ngrid), dtype=data.dtype)
    
    i_coords = indices[:, 0]
    j_coords = indices[:, 1]
    
    grid = grid.at[..., i_coords, j_coords].set(data)
    grid = grid.at[..., -i_coords, -j_coords].set(data.conj())
    grid = grid.at[..., 0, 0].set(1.0)
    return grid

@partial(jax.jit, static_argnames=['ngrid'])
def grid_data_array(data, indices, ngrid):
    """
    Grid baseline data onto a 2D grid using indices.
    
    Args:
        data: Array of shape (..., nbls) containing baseline data
        indices: Array of shape (nbls, 2) containing grid positions for each baseline
        ngrid: Size of the square grid
    
    Returns:
        Gridded array of shape (..., ngrid, ngrid)
    """
    batch_shape = data.shape[:-1]
    grid = jnp.zeros(batch_shape + (ngrid, ngrid), dtype=data.dtype)
    
    i_coords = indices[:, 0]
    j_coords = indices[:, 1]
    
    grid = grid.at[..., i_coords, j_coords].set(data)
    grid = grid.at[..., -i_coords, -j_coords].set(data.conj())
    return grid

@jax.jit
def degrid_array(gridded_data, indices):
    """
    Extract baseline data from gridded array using indices.
    
    Args:
        gridded_data: Array of shape (..., ngrid, ngrid)
        indices: Array of shape (nbls, 2) containing grid positions
    
    Returns:
        Baseline data of shape (..., nbls)
    """
    i_coords = indices[:, 0]
    j_coords = indices[:, 1]
    
    return gridded_data[..., i_coords, j_coords]

@jax.jit
def fft_deconvolve(coupling: jnp.ndarray, data_fft: jnp.ndarray) -> jnp.ndarray:
    """
    Deconvolve visibilities using the provided parameters and FFT of the data.

    TODO: Should be consider using a regularized deconvolution method?

    Parameters:
        coupling : jnp.ndarray
            Coupling parameters for the deconvolution of shape (1, nfreqs, ngrid, ngrid).
        data_fft : jnp.ndarray
            2D-FFT of the gridded data along the grid axes. 

    Returns:
        jnp.ndarray: Deconvolved visibilities
    """
    preturbed_beam = jnp.fft.fft2(coupling)
    div = data_fft * (1 / preturbed_beam)
    deconvolved = jnp.fft.ifft2(div)
    return deconvolved

@partial(jax.jit, static_argnames=['ngrid'])
def fft_deconvolve_low_memory(
    data: jnp.ndarray, 
    coupling: jnp.ndarray, 
    indices: jnp.ndarray, 
    coupling_indices: jnp.ndarray, 
    fit_indices: jnp.ndarray, 
    ngrid: int
) -> jnp.ndarray:
    """
    Memory-efficient FFT convolution using scan across frequency.
    Processes one frequency at a time while handling all time steps together.
    
    TODO: There are lots of indices here that could be cleaned up.

    Args:
        data: 
            Array of shape (ntimes, nfreqs, nbls)
        coupling: 
            Array of shape (1, nfreqs, nbls)
        indices: 
            Array of shape (nbls, 2) that specifies the grid positions for each baseline.
        coupling_indices: 
            Array of shape (ncoupling, 2) that specifies the coupling parameters.
        fit_indices:
            Array of shape (n_fit_baselines, 2) that specifies the baselines used in the fitting process.
        ngrid: 
            Grid size
    
    Returns:
        Deconvolved result of shape (ntimes, nfreqs, nbls)
    """
    ntimes, nfreqs, nbls = data.shape
    coupling_squeezed = coupling[0]  # (nfreqs, nbls)
    
    def scan_frequency(carry, inputs):
        """
        Process a single frequency across all time steps.
        
        Args:
            carry: Unused (scan requires carry argument)
            inputs: Tuple of (data, coupling) where:
                data: Array of shape (ntimes, nbls)
                coupling: Array of shape (nbls,)
        
        Returns:
            carry: Unused
            result: Deconvolved result for this frequency (ntimes, nbls)
        """
        data, coupling = inputs
        
        # Grid both arrays for this frequency
        data_grid = grid_data_array(data, indices, ngrid)  # (ntimes, ngrid, ngrid)
        coupling_grid = grid_coupling_array(coupling, coupling_indices, ngrid)  # (ngrid, ngrid)
        
        # Take FFTs
        data_fft = jnp.fft.fft2(data_grid)  # (ntimes, ngrid, ngrid)
        coupling_fft = jnp.fft.fft2(coupling_grid)  # (ngrid, ngrid)

        # Convolve (broadcast coupling across time)
        # div = data_fft * (1 / coupling_fft[None, ...])  # (ntimes, ngrid, ngrid)
        div = data_fft * (1 / coupling_fft)
        deconvolved = jnp.fft.ifft2(div)  # (ntimes, ngrid, ngrid)

        # Degrid back to baseline format
        result = degrid_array(deconvolved, fit_indices)  # (ntimes, nbls)
        
        return carry, result
    
    # Transpose data to (nfreqs, ntimes, nbls) for scanning
    data_tranposed = jnp.transpose(data, (1, 0, 2))
    
    # Use scan to process each frequency sequentially
    _, results = jax.lax.scan(
        scan_frequency, None, (data_tranposed, coupling_squeezed)
    )
    
    # Transpose back to (ntimes, nfreqs, nbls)
    return jnp.transpose(results, (1, 0, 2))

@jax.jit
def deconv_loss_function(
    parameters: dict,
    data_fft: jnp.ndarray,
    noise: jnp.ndarray,
    coupling_idx: jnp.ndarray,
    data_idx: jnp.ndarray,
    lambda_reg: float = 0.0,
) -> jnp.ndarray:
    """
    Compute the loss function for deconvolution.
    
    Parameters:
    -----------
        parameters : dict
            Parameters for the deconvolution. 
        data_fft : jnp.ndarray
            FFT of the input data. (ntimes, nfreqs, ngrid, ngrid)
        noise : jnp.ndarray
            Noise variance for the data.
        coupling_idx : jnp.ndarray
            Indices for the data to be deconvolved.
        data_idx : jnp.ndarray
            Indices for the coupling parameters in the grid.
        window : jnp.ndarray
            Window function applied to the data.
        lambda_reg : float, optional
            Regularization parameter for parameter sparsity (default: 0.0).

    Returns:
    --------
        jnp.ndarray
            Computed loss value.
    """
    # Extract coupling parameters
    coupling_params = parameters['coupling']

    # Initialize coupling array with zeros and add the coupling parameters
    coupling = jnp.zeros((1,) + data_fft.shape[1:], dtype=complex)
    coupling = coupling.at[:, :, coupling_idx[:, 0], coupling_idx[:, 1]].add(
        coupling_params
    )
    coupling = coupling.at[:, :, -coupling_idx[:, 0], -coupling_idx[:, 1]].add(
        jnp.conj(coupling_params)
    )
    coupling = coupling.at[:, :, 0, 0].add(1)
    
    # Deconvolve coupling data using FFT
    data_deconv = fft_deconvolve(
        coupling=coupling, 
        data_fft=data_fft
    )

    # Extract deconvolved data for the specified indices, applying the window function, and compute the FFT
    data_deconv_fft = jnp.fft.fft2(
        data_deconv[:, :, data_idx[:, 0], data_idx[:, 1]],
        axes=(0, 1),
        norm="ortho" # Normalize the FFT
    )
    
    # Minimize the size of the coupling parameters
    param_sparsity_term = jnp.sum(
        jnp.abs(coupling_params) ** 2
    )

    # Compute the delay fringe sparsity
    delay_fringe_sparsity = jnp.mean(
        _scaled_log_1p_normalized(
            jnp.abs(data_deconv_fft) ** 2 / noise
        )
    )
    
    # Combine the loss components
    total_loss = (
        delay_fringe_sparsity +
        lambda_reg * param_sparsity_term
    )
    
    return total_loss

@partial(jax.jit, static_argnames=['ngrid'])
def deconv_loss_function_batched(
    parameters: dict,
    data: jnp.ndarray,
    noise: jnp.ndarray,
    coupling_idx: jnp.ndarray,
    data_idx: jnp.ndarray,
    fit_idx: jnp.ndarray,
    ngrid: int,
    lambda_reg: float=0.0,
):
    """
    """
    data_deconv = fft_deconvolve_low_memory(
        data, 
        parameters['coupling'], 
        data_idx, 
        coupling_idx, 
        fit_idx,
        ngrid
    )

    # Extract coupling parameters
    coupling_params = parameters['coupling']
    
    # Extract deconvolved data for the specified indices, applying the window function, and compute the FFT
    data_deconv_fft = jnp.fft.fft(
        data_deconv,
        axis=1, 
        norm="ortho"
    )
    effective_var = jnp.sqrt(noise[::2] * noise[1::2])
    #data_fft_avg = jnp.mean(
    #    (data_deconv_fft[::2] * data_deconv_fft[1::2].conj()) * effective_var ** -1, axis=0
    #) / jnp.mean(effective_var ** -1, axis=0)
    #avg_var = jnp.sum((effective_var ** 2) ** -1, axis=0) ** -0.5 / jnp.sqrt(effective_var.shape[0])

    # Minimize the size of the coupling parameters
    param_sparsity_term = jnp.sum(
        jnp.abs(coupling_params) ** 2
    )
    
    delay_fringe_sparsity = jnp.mean(
        _scaled_log_1p_normalized(
            jnp.abs(data_deconv_fft[::2] * data_deconv_fft[1::2].conj()) / effective_var
        ) / effective_var
    )
    
    # Combine the loss components
    total_loss = (
        delay_fringe_sparsity +
        lambda_reg * param_sparsity_term
    )
    
    return total_loss

def estimate_windowed_noise_variance(
    data: DataContainer, 
    flags: DataContainer,
    nsamples: DataContainer,
    time_slice: slice, 
    freq_slice: slice, 
    window_function: str = "tukey", 
    axes: tuple = (0, 1),
    dt=10.,
    df=122e3,
) -> Dict[Tuple[int, int, str], jnp.ndarray]:
    """
    Estimate the windowed noise variance for the coupling deconvolution. Used to set the
    noise scale for the coupling deconvolution.

    Parameters:
    -----------
        data : DataContainer
            The data container containing the visibilities to be deconvolved.
            Autocorrelations must be included in the data container.
        nsamples : DataContainer
            The number of samples for each visibility in the data container.
        time_slice : slice
            Time slice for the data.
        freq_slice : slice
            Frequency slice for the data.
        window_function : str, optional
            The window function to apply (default: "tukey").

    Returns:
    --------
        noise_scale : jnp.ndarray
            The estimated noise scale for the coupling deconvolution.
    """
    if 0 in axes:
        time_window = dspec.gen_window(
            window_function,
            data.times[time_slice].size,
        )
    else:
        time_window = np.ones(data.times[time_slice].size)

    # Get the window function for the frequency and time slices
    if 1 in axes:
        freq_window = dspec.gen_window(
            window_function,
            data.freqs[freq_slice].size,
        )
    else:
        freq_window = np.ones(data.freqs[freq_slice].size)

    window = np.outer(time_window, freq_window)
    
    # Calculate the equivalent noise bandwidth for the window function
    enbw = np.mean(window ** 2) / np.mean(window) ** 2
    
    # Dictionary to hold the noise variance for each baseline
    noise_var = {}

    # Extract the relevant data from the data container
    for key in data:
        ant1, ant2 = utils.split_bl(key)
        ap1, ap2 = utils.split_pol(key[-1])

        auto_ant1 = (ant1[0], ant1[0], utils.join_pol(ap1, ap1))
        auto_ant2 = (ant2[0], ant2[0], utils.join_pol(ap2, ap2))

        auto1, auto2 = (
            data[auto_ant1][time_slice][:, freq_slice], 
            data[auto_ant2][time_slice][:, freq_slice]
        )

        # Calculate the time and frequency intervals
        # dt = np.diff(data.times)[0] * 3600 * 24  # Convert to seconds
        # df = np.diff(data.freqs)[0]

        # nsamples & flags for this baseline
        ns = nsamples[key][time_slice][:, freq_slice]
        fl = flags[key][time_slice][:, freq_slice]

        # mask out any cells where ns==0 or flagged=True
        valid = (ns > 0) & (~fl)

        variance = (
            np.abs(auto1) * np.abs(auto2) / (ns * dt * df)
        )
        variance = np.where(valid, variance, np.nan)

        # Calculate the noise scale for the autocorrelations
        noise_var[key] = np.nanmean(
                window ** 2 * variance, 
                axis=axes, 
                keepdims=True
        ) / np.nanmean(window, axis=axes, keepdims=True) ** 2  # Average over time and frequency

    # Convert the noise variance to a DataContainer
    noise_var = DataContainer(noise_var)

    return noise_var, window

def fit_coupling_redundantly_averaged(
    coupling_parameters: jnp.ndarray, 
    grid_data: jnp.ndarray, 
    noise: jnp.ndarray, 
    coupling_idx: jnp.ndarray, 
    data_idx: jnp.ndarray,
    window: jnp.ndarray,
    fit_idx: jnp.ndarray = None,
    ngrid: int = None,
    compressed: bool = False,
    maxiter: int = 100, 
    use_LBFGS: bool = True, 
    optimizer: optax.GradientTransformation = None,
    tol: float = 1e-6,
    history_size: int = 10,
    linesearch: str = "zoom",
    lambda_reg: float = 1e-3,
    verbose: bool = False,
    **kwargs
) -> Tuple[dict, Union[dict, List[float]]]:
    """
    Optimize parameters using either L-BFGS or a custom optimizer.

    This function supports two optimization strategies:
    1. L-BFGS (recommended for most cases)
    2. Custom optimizer with manual gradient descent

    Parameters:
    -----------
        coupling_parameters : jnp.ndarray
            Initial parameters to be optimized. Should have shape (nparams, nfreqs)
        grid_data : jnp.ndarray
            Input grid data for optimization
        noise : jnp.ndarray
            Noise variance for the data, used to scale the coupling parameters
        coupling_idx : jnp.ndarray
            Indices for the coupling parameters in the grid of shape (ncoupling_antpairs,
            2).
        data_idx : jnp.ndarray
            Indices for the data to be deconvolved in the grid of shape (ndata_ant
        window : jnp.ndarray
            Window function applied to the data of shape (ntimes, nfreqs).
        lamb : float, optional
            Regularization parameter (default: 1e-3)
        maxiter : int, optional
            Maximum number of iterations (default: 100)
        use_LBFGS : bool, optional
            Whether to use L-BFGS optimizer (default: True)
        optimizer : optax.GradientTransformation, optional
            Custom optimizer if not using L-BFGS
        tol : float, optional
            Tolerance for optimization convergence (default: 1e-6)

    Returns:
    --------
        Tuple[dict, Union[dict, List[float]]]
            Optimized parameters and metadata/loss history
    """
    # Check if the data is compressed
    if compressed and ngrid is None:
        raise ValueError("ngrid must be provided if data is compressed")
    if compressed and fit_idx is None:
        raise ValueError("fit_idx must be provided if data is compressed")
    
    # TODO: Should do some validation checks of the inputs here
    model_params = {
        'coupling': coupling_parameters
    }
    
    # Prepare the optimization to handle compressed data
    if not compressed:
        data_fft = jnp.fft.fft2(grid_data * window[..., None, None])
        loss_function = partial(
            deconv_loss_function,
            data_fft=data_fft,
            noise=noise,
            coupling_idx=coupling_idx,
            data_idx=data_idx,
            lambda_reg=lambda_reg,
        )
    else:
        data = grid_data * window[..., None]
        loss_function = partial(
            deconv_loss_function_batched,
            data=data,
            noise=noise,
            coupling_idx=coupling_idx,
            data_idx=data_idx,
            ngrid=int(ngrid),
            fit_idx=fit_idx,
            lambda_reg=lambda_reg,
        )
    
    # Check if the user wants to use L-BFGS or a custom optimizer
    if use_LBFGS:        
        # Use L-BFGS optimizer
        solver = jaxopt.LBFGS(
            fun=loss_function, 
            tol=tol, 
            maxiter=maxiter,
            verbose=verbose,
            history_size=history_size,
            linesearch=linesearch,
            **kwargs,
        )

        solved_parameters, meta = solver.run(
            model_params,  
        )

        return solved_parameters, meta

    else:
        # Custom optimizer gradient descent
        if optimizer is None:
            raise ValueError("Must provide an optimizer when use_LBFGS is False")
        
        opt_state = optimizer.init(model_params)
        loss_history = []
        
        for nit in tqdm.tqdm(range(maxiter), desc="Optimization Progress"):
            # Compute loss and gradients
            loss_value, grads = jax.value_and_grad(loss_function)(
                model_params,
            )
            
            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state)
            model_params = optax.apply_updates(model_params, updates)
            
            # Track loss history
            loss_history.append(loss_value)
            
            # Optional early stopping
            if len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) < tol:
                if verbose:
                    print(f"Converged after {nit+1} iterations")
                break
        
        return model_params, loss_history
    

class RedUVCoupling:
    """
    Class for deconvolving visibilities using a coupling grid.

    Sketch:
        1. Take in a coupling grid which handles the antenna positions.
           The coupling grid is used to project the antenna positions onto a 2D grid.
        2. Use the coupling grid to grid the visibilities. Also requires the region of interest.
        3. Fit the coupling parameters using the deconvolution method.
        4. Use the fitted coupling parameters to deconvolve the visibilities.
    """
    def __init__(self, coupling_grid: RedundantCouplingManager):
        """
        Initialize the CouplingDeconvolution class.

        Parameters:
        -----------
            coupling_grid : RedundantCouplingManager
                The coupling grid for the antenna positions.
        """
        self.coupling_grid = coupling_grid

        # TODO: Also need to store the coupling parameters
        
    def apply(
            self, 
            data: DataContainer,
            first_order: bool=False,
            multi_path: bool=False,
            inplace: bool=False,
        ):
        """
        TODO: now have UVCoupling class that handles deconvolution.
              Should I make a RedUVCoupling class that inherits from UVCoupling
              and uses FFTs instead of matrix multiplication?

        Deconvolve the visibilities using the fitted coupling parameters.
        
        Parameters:
        -----------
            time_slice : slice
                Time slice for the data.
            freq_slice : slice
                Frequency slice for the data.
        
        Returns:
        --------
            deconvolved_visibilities : array-like
                The deconvolved visibilities.
        """
        if not inplace:
            data = copy.deepcopy(data)

        # Placeholder for actual data extraction logic
        for pol, pi in enumerate(data.pols):
            for ti, time in enumerate(data.times):
                # Extract the visibility data for the current time and frequency
                grid_data = self.coupling_grid.build_data_grid(
                    data=data,
                    time_slice=slice(ti, ti + 1),
                    pol=pi,
                )
                coupling_params = self.coupling_grid.select_coupling(
                    time_slice=slice(ti, ti + 1),
                    pol=pi,
                )
                for fi, freq in enumerate(data.freqs):
                    # TODO: Apply the coupling deconvolution logic here
                    # This is a placeholder for the actual deconvolution logic
                    # deconvolved_visibilities = self.coupling_grid.deconvolve(vis_data)
                    
                    # If inplace is True, modify the data in place
                    if inplace:
                        data[(time, freq, pi)] = deconvolved_visibilities
                    else:
                        return deconvolved_visibilities

    def to_uvcoupling(self) -> UVMutualCoupling:
        """
        Convert the coupling grid to a UVCoUVMutualCoupling object, where the coupling parameters
        have been expanded to an antenna-by-antenna coupling matrix.

        TODO: Implement the conversion logic to UVMutualCoupling.

        Returns:
        --------
            uvm : UVMutualCoupling
                The UVMutualCoupling object containing the coupling parameters.
        """
        # Placeholder for actual conversion logic
        raise NotImplementedError("Conversion to UVMutualCoupling not implemented yet.")
