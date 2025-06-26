import numpy as np
from typing import Tuple, Union, List

# Jax libraries
import jax
import tqdm
import jaxopt
import optax
from jax import numpy as jnp
jax.config.update("jax_enable_x64", True)

from hera_filters import dspec
from hera_cal.datacontainer import DataContainer
"""
Outstanding Issues:
1. The `project_baselines_to_grid` function needs to be integrated with the coupling grid
   to ensure that the antenna positions are correctly projected onto the grid.
2. The `CouplingGrid` class needs to be fully implemented to handle the coupling parameters
   and provide methods for selecting and setting coupling values.
3. Need to decide if the 

"""


class CouplingGrid:
    def __init__(self, antpos, include_autos: bool = False):
        """
        Parameters:
        -----------
            antpos : dictionary
                Antenna positions of the active antennas in the array
            include_autos : bool, default=False
                Flag on whether to include auto correlations in visibility grid
        """
        self.antpos = antpos
        
        # Initialize the coupling grid
        self.prepare_coordinates(
            include_autos=include_autos
        )
        
    def prepare_coordinates(self, include_autos: bool = False):
        """
        Setup

        Parameters:
        -----------
        include_autos : bool, default=False
            Flag on whether to include auto correlations in visibility grid.
        """
        project_baselines_to_grid()
        self.coupling = {}
        
    def select_coupling(
        self, 
        ubl_keys: list[Tuple]=None, 
        max_bl_cut: float=None, 
        max_ew_length: float=None,
        max_ns_length: float=None
    ):
        """
        Select the coupling 

        Parameters:
        -----------
        ubl_keys : list of tuples, optional
            List of unique baseline keys to select from the coupling grid.
            If None, all baselines are selected.
        max_bl_cut : float, optional
            Maximum baseline cut-off distance for selection.
        max_ew_length : float, optional
            Maximum east-west length for selection.
        max_ns_length : float, optional
            Maximum north-south length for selection.

        Returns:
        --------
        coupling_values : array_like
            The selected coupling values based on the provided parameters.
        index : list
            The indices of the selected coupling values in the original coupling grid.
        """
        if ubl_keys is None:
            ubl_keys = []
        if max_bl_cut is None:
            max_bl_cut = np.inf
        if max_ew_length is None:
            max_ew_length = np.inf
        if max_ns_length is None:
            max_ns_length = np.inf

        # Filter the used coupling parameters based on the provided filters
        # This is a placeholder for the actual selection logic
        pass

    def set_coupling(self, antpairs, coupling_values):
        """
        Overwrite the coupling values stored in coupling array
        """
        pass

def project_baselines_to_grid(antpairs, antpos, ew_pair=(0, 1), ns_pair=(0, 11), ratio: int = 3):
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

def build_coupling_grid(antpos, uvw_grid, ratio: int = 1):
    """
    Builds the coupling grid for the antenna positions.

    Parameters
    ----------
    antpos : array_like
        The antenna positions to be projected.
    uvw_grid : array_like
        The UVW coordinates of the grid.
    ratio : int
        The ratio of the antenna positions to be projected.
    
    Returns
    -------
    tuple
        The coupling grid for the antenna positions.
    """
    # Get the antenna pairs
    antpair = np.array([antpos[i] for i in range(len(antpos))])
    # Project the antenna positions onto a grid defined by the ratio
    uvw_grid = project_baselines_to_grid(antpair, antpos, ratio)
    # Build the coupling grid
    coupling_grid = np.zeros((len(antpair), len(uvw_grid)))
    for i in range(len(antpair)):
        coupling_grid[i] = uvw_grid[i]

    return coupling_grid

@jax.jit
def _scaled_log_1p_normalized(data):
    """
    Computes the scaled log(max(x - 1, 0)) function.

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
    return jnp.log1p(jnp.maximum(data - 1, 0))

@jax.jit
def deconvolve_visibilities(coupling: jnp.ndarray, data_fft: jnp.ndarray) -> jnp.ndarray:
    """
    Deconvolve visibilities using the provided parameters and FFT of the data.

    Parameters:
        coupling : jnp.ndarray
            Coupling parameters for the deconvolution of shape (1, nfreqs, ngrid, ngrid).
        data_fft : jnp.ndarray
            2D-FFT of the gridded data along the grid axes. 

    Returns:
        jnp.ndarray: Deconvolved visibilities
    """
    model = jnp.fft.ifft2(data_fft / jnp.fft.fft2(coupling))
    return model

@jax.jit
def deconv_loss_function(
    parameters: dict,
    data_fft: jnp.ndarray,
    noise: jnp.ndarray,
    idx: jnp.ndarray,
    window: jnp.ndarray,
    alpha: float = 1e-3,
    lambda_reg: float = 1e-3,
) -> jnp.ndarray:
    """
    Compute the loss function for deconvolution.
    
    Parameters:
    -----------
        parameters : dict
            Parameters for the deconvolution. 
        data_fft : jnp.ndarray
            FFT of the input data.
        mask : jnp.ndarray
            Mask applied during optimization.
        window : jnp.ndarray
            Window function applied to the data.
        alpha : float, optional
            Regularization parameter (default: 1e-3).
        lambda_reg : float, optional
            Regularization parameter for parameter sparsity (default: 1e-3).

    Returns:
    --------
        jnp.ndarray
            Computed loss value.
    """
    # Extract coupling parameters
    coupling_params = parameters['coupling']

    # Initialize coupling array with zeros and add the coupling parameters
    coupling = jnp.zeros((1,) + data_fft.shape[1:], dtype=complex)
    coupling = coupling.at[:, :, idx[:, 0], idx[:, 1]].add(coupling_params)
    
    # Deconvolve coupling data using FFT
    data_deconv = deconvolve_visibilities(
        coupling=coupling, 
        data_fft=data_fft
    )

    # Extract deconvolved data for the specified indices, applying the window function, and compute the FFT
    data_deconv_fft = jnp.fft.fft2(
        data_deconv[:, :, idx[:, 0], idx[:, 1]] * window, 
        axes=(0, 1), 
        norm='ortho'
    )
    
    # Extract coupling parameters and compute the loss
    primary_coupling_params = coupling_params[..., 0]
    regularization_term = jnp.mean(
        jnp.square(
            jnp.abs(primary_coupling_params - 1.0)
            )
    )

    # Minimize the size of the coupling parameters
    param_sparsity_term = jnp.sum(
        jnp.abs(coupling_params[..., 1:]) ** 2
    )

    # Compute the delay fringe sparsity
    delay_fringe_sparsity = jnp.mean(
        _scaled_log_1p_normalized(
            jnp.abs(data_deconv_fft) / noise
        )
    )
    
    # Combine the loss components
    total_loss = (
        delay_fringe_sparsity +
        alpha * regularization_term +
        lambda_reg * param_sparsity_term
    )
    
    return total_loss


def fit_coupling_redundantly_averaged(
    parameters: dict, 
    grid_data: jnp.ndarray, 
    noise: jnp.ndarray, 
    idx: jnp.ndarray, 
    window: jnp.ndarray, 
    alpha: float = 1e-3, 
    maxiter: int = 100, 
    use_LBFGS: bool = True, 
    optimizer: optax.GradientTransformation = None,
    tol: float = 1e-6,
    history_size: int = 10,
    linesearch: str = "zoom",
    lambda_reg: float = 1e-3,
    verbose: bool = False
) -> Tuple[dict, Union[dict, List[float]]]:
    """
    Optimize parameters using either L-BFGS or a custom optimizer.

    This function supports two optimization strategies:
    1. L-BFGS (recommended for most cases)
    2. Custom optimizer with manual gradient descent

    Parameters:
    -----------
        parameters : dict
            Initial parameters to be optimized
        grid_data : jnp.ndarray
            Input grid data for optimization
        mask : jnp.ndarray
            Mask applied during optimization
        window : jnp.ndarray
            Window function applied to the data
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
    # Compute FFT of input data
    data_fft = jnp.fft.fft2(grid_data)
    
    # Validate inputs
    if use_LBFGS:        
        # Use L-BFGS optimizer
        solver = jaxopt.LBFGS(
            fun=deconv_loss_function, 
            tol=tol, 
            maxiter=maxiter,
            verbose=verbose,
            history_size=history_size,
            linesearch=linesearch,
        )

        solved_params, meta = solver.run(
            parameters,  
            data_fft=data_fft, 
            noise=noise,
            idx=idx, 
            window=window, 
            alpha=alpha,
            lambda_reg=lambda_reg,
        )

        return solved_params, meta

    else:
        # Custom optimizer gradient descent
        if optimizer is None:
            raise ValueError("Must provide an optimizer when use_LBFGS is False")
        
        opt_state = optimizer.init(parameters)
        loss_history = []
        
        for nit in tqdm.tqdm(range(maxiter), desc="Optimization Progress"):
            # Compute loss and gradients
            loss_value, grads = jax.value_and_grad(deconv_loss_function)(
                parameters,  
                data_fft=data_fft, 
                noise=noise,
                idx=idx, 
                window=window, 
                alpha=alpha,
                lambda_reg=lambda_reg,
            )
            
            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state)
            parameters = optax.apply_updates(parameters, updates)
            
            # Track loss history
            loss_history.append(loss_value)
            
            # Optional early stopping
            if len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) < tol:
                if verbose:
                    print(f"Converged after {nit+1} iterations")
                break
        
        return parameters, loss_history
    
class CouplingDeconvolution:
    """
    Class for deconvolving visibilities using a coupling grid.

    Sketch:
        1. Take in a coupling grid which handles the antenna positions.
           The coupling grid is used to project the antenna positions onto a 2D grid.
        2. Use the coupling grid to grid the visibilities. Also requires the region of interest.
        3. Fit the coupling parameters using the deconvolution method.
        4. Use the fitted coupling parameters to deconvolve the visibilities.
    """
    def __init__(self, coupling_grid: CouplingGrid):
        """
        Initialize the CouplingDeconvolution class.

        Parameters:
        -----------
            coupling_grid : CouplingGrid
                The coupling grid for the antenna positions.
        """
        self.coupling_grid = coupling_grid

    def estimate_noise_scale(
            self, 
            data: DataContainer, 
            time_slice: slice, 
            freq_slice: slice, 
            window_function: str = "tukey", 
        ):
        """
        Estimate the noise scale for the coupling deconvolution. Used to set the
        noise scale for the coupling deconvolution.

        Parameters:
        -----------
            data : DataContainer
                The data container containing the visibilities to be deconvolved.
                Autocorrelations must be included in the data container.
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
        # Get the window function for the frequency and time slices
        freq_window = dspec.get_window(
            window_function,
            data.freqs[freq_slice].size,
        )
        time_window = dspec.get_window(
            window_function,
            data.times[time_slice].size,
        )
        window = jnp.outer(time_window, freq_window)
        
        # Calculate the equivalent noise bandwidth for the window function
        enbw = ...
        
        # Extract the relevant data from the data container
        for key in data:
            pass

    def fit_coupling_parameters(
            self, 
            data: DataContainer,
            time_slice: slice,
            freq_slice: slice, 
            window_function: str="tukey", 
            use_LBFGS: bool=True,
            maxiter: int=100
        ) -> dict:
        """
        Fit for time-independent coupling parameters using FFTs of the gridded data.

        Parameters:
        -----------
        data : DataContainer
            The data container containing the visibilities to be deconvolved.
        time_slice : slice
            Time slice for the data.
        freq_slice : slice
            Frequency slice for the data.
        window_function : str, optional
            The window function to apply (default: "tukey").
        use_LBFGS : bool, optional
            Whether to use L-BFGS optimizer (default: True).
        
        Returns:
        --------
        fit_parameters : dict
            The fitted coupling parameters.
        """
        # Extract the relevant data from the data container
        # TODO: Get data using the gridding code

        # Extract the relevant data from the coupling grid
        coupling_values, index = self.coupling_grid.select_coupling(
            ubl_keys=None, 
            max_bl_cut=None, 
            max_ew_length=None, 
            max_ns_length=None
        )

        fit_parameters = fit_coupling_redundantly_averaged(
            coupling_values,
            grid_data=None,  # Placeholder for actual grid data
            mask=None,  # Placeholder for actual mask
            window=None,  # Placeholder for actual window
            maxiter=maxiter,
            use_LBFGS=use_LBFGS,
        )

        # TODO: Store fit parameters in a more structured way
        #       ideally something like UVCoupling class, maybe
        #       RedUVCoupling class that inherits from UVCoupling
        #       and uses FFTs instead of matrix multiplication.

        return fit_parameters
    
    def deconvolve_visibilities(
            self, 
            data: DataContainer,
            time_slice: slice, 
            freq_slice: slice
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
        # Placeholder for actual data extraction logic
        pass