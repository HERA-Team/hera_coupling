import pytest
import numpy as np
from copy import deepcopy

# Import hera modules
from hera_coupling import UVMutualCoupling
from hera_coupling.coupling import _extract_data_matrix
from hera_cal.datacontainer import DataContainer


class TestApplyCoupling:
	def setup_method(self):
		# Setup the test environment
		rng = np.random.default_rng(seed=42)

		# Initialize the coupling with some dummy data
		self.nants = 10
		self.nfreqs = 5
		self.ntimes = 2
		self.pols = ['ee', 'nn']
		self.npols = len(self.pols)

		self.antpos = {
			i: np.array([i, 0, 0])
			for i in range(self.nants)
		}
		self.freqs = np.linspace(100e6, 200e6, self.nfreqs)
		self.times = np.linspace(2458168.1, 2458168.3, self.ntimes)

		shape = (2, self.nants, self.nants, 1, self.nfreqs)
		coupling = rng.normal(0, 1e-1, shape) + 0j
		coupling = (np.swapaxes(coupling, 1, 2).conj() + coupling) / 2
		self.uvm = UVMutualCoupling(
			coupling=coupling,
			antpos=self.antpos,
			freqs=self.freqs,
			pols=self.pols,
		)

		shape = (2, self.nants, self.nants, self.ntimes, self.nfreqs)
		coupling = rng.normal(0, 1e-1, shape) + 0j
		coupling = (np.swapaxes(coupling, 1, 2).conj() + coupling) / 2
		self.uvm_time_varying = UVMutualCoupling(
			coupling=coupling,
			antpos=self.antpos,
			freqs=self.freqs,
			pols=self.pols,
			times=self.times,
		)

		self.data = {
			(i, j, pol): rng.normal(0, 1, size=(self.ntimes, self.nfreqs)) + 0j
			for i in range(self.nants)
			for j in range(i)
			for pol in self.pols
		}

		for pol in self.pols:
			d = rng.uniform(0, 1, size=(self.ntimes, self.nfreqs)) + 0j
			for i in range(len(self.antpos)):
				self.data[(i, i, pol)] = np.copy(d)
		
		self.data = DataContainer(self.data)
		self.data.antpos = self.antpos
		self.data.ants = list(self.antpos.keys())
		self.data.freqs = self.freqs
		self.data.times = self.times
		self.data.pols = self.pols

	@pytest.mark.parametrize("first_order", [True, False])
	@pytest.mark.parametrize("multi_path", [True, False])
	@pytest.mark.parametrize("use_numpy", [True, False])
	def test_invert_coupling(self, first_order, multi_path, use_numpy):
		if use_numpy:
			data = np.zeros(
				(self.npols, self.nants, self.nants, self.ntimes, self.nfreqs), 
				dtype=np.complex128
			)
			for pi, pol in enumerate(self.pols):
				for ti in range(self.ntimes):
					temp = _extract_data_matrix(
						self.data,
						self.antpos,
						time_idx=ti,
						pol=pol,
					)
					data[pi, :, :, ti, :] = np.transpose(temp, (1, 2, 0))
		else:
			data = self.data

		uvm = deepcopy(self.uvm)
		uvm_time_varying = deepcopy(self.uvm_time_varying)

		# Invert the coupling
		coupled_data = uvm.apply(
			data, 
			forward=True,
			first_order=first_order,
			multi_path=multi_path,
			inplace=False
		)

		uncoupled_data = uvm.apply(
			coupled_data, 
			forward=False,
			first_order=first_order,
			multi_path=multi_path,
			inplace=False
		)

		# Check type of uncoupled_data
		if use_numpy:
			assert isinstance(uncoupled_data, np.ndarray), "Uncoupled data should be a numpy array"
			assert isinstance(coupled_data, np.ndarray), "Coupled data should be a numpy array"
		else:
			assert isinstance(uncoupled_data, DataContainer), "Uncoupled data should be a DataContainer"
			assert isinstance(coupled_data, DataContainer), "Coupled data should be a DataContainer"

		if use_numpy:
			assert np.allclose(uncoupled_data, data, atol=1e-10), f"Data mismatch"
		else:
			for key in uncoupled_data:
				# Check if the uncoupled data matches the original data
				assert np.allclose(uncoupled_data[key], data[key], atol=1e-10), f"Data mismatch for key {key}"

		coupled_data = uvm_time_varying.apply(
			data, 
			forward=True,
			first_order=first_order,
			multi_path=multi_path,
			inplace=False
		)

		uncoupled_data = uvm_time_varying.apply(
			coupled_data, 
			forward=False,
			first_order=first_order,
			multi_path=multi_path,
			inplace=False
		)

		# Check type of uncoupled_data
		if use_numpy:
			assert isinstance(uncoupled_data, np.ndarray), "Uncoupled data should be a numpy array"
			assert isinstance(coupled_data, np.ndarray), "Coupled data should be a numpy array"
		else:
			assert isinstance(uncoupled_data, DataContainer), "Uncoupled data should be a DataContainer"
			assert isinstance(coupled_data, DataContainer), "Coupled data should be a DataContainer"

		if use_numpy:
			assert np.allclose(uncoupled_data, data, atol=1e-10), f"Data mismatch"
		else:
			for key in uncoupled_data:
				# Check if the uncoupled data matches the original data
				assert np.allclose(uncoupled_data[key], data[key], atol=1e-10), f"Data mismatch for key {key}"

	@pytest.mark.parametrize("first_order", [True, False])
	@pytest.mark.parametrize("multi_path", [True, False])
	def test_inverse(self, first_order, multi_path):
		uvm = deepcopy(self.uvm)
		uvm_time_varying = deepcopy(self.uvm_time_varying)
		
		# Invert the coupling
		uvm.invert(first_order=first_order, multi_path=multi_path)
		uvm_time_varying.invert(first_order=first_order, multi_path=multi_path)

		if not first_order:
			# Check if the coupling matrix is Hermitian
			assert uvm.is_inverted, "Coupling should be inverted"
			assert uvm.inverse_coupling.shape == uvm.coupling.shape, "Inverse coupling shape mismatch" 
			assert uvm_time_varying.is_inverted, "Coupling should be inverted"
			assert uvm_time_varying.inverse_coupling.shape == uvm_time_varying.coupling.shape, "Inverse coupling shape mismatch" 
		else:
			# For first order, Sylvester solver is used, so 
			assert not uvm.is_inverted, "First order coupling should not be inverted"
			assert not hasattr(uvm, "inverse_coupling"), "First order coupling should not have inverse_coupling attribute"
			assert not uvm_time_varying.is_inverted, "First order coupling should not be inverted"
			assert not hasattr(uvm_time_varying, "inverse_coupling"), "First order coupling should not have inverse_coupling attribute"

	def test_coupling_io(self, tmp_path):
		# Test saving and loading the coupling
		file_path = tmp_path / "test_coupling.hdf5"
		self.uvm.write_coupling(file_path)
	
		with pytest.raises(FileExistsError):
			# Attempt to overwrite the file without overwrite flag
			self.uvm.write_coupling(file_path, clobber=False)

		loaded_uvm = UVMutualCoupling.read_coupling(file_path)
		with pytest.raises(FileNotFoundError):
			# Attempt to read a non-existent file
			UVMutualCoupling.read_coupling(tmp_path / "non_existent_file.hdf5")
		
		assert isinstance(loaded_uvm, UVMutualCoupling), "Loaded object should be a uvm"
		assert np.allclose(loaded_uvm.coupling, self.uvm.coupling), "Coupling matrices should match"
		for k in self.uvm.antpos:
			assert np.allclose(loaded_uvm.antpos[k], self.uvm.antpos[k]), f"Antenna position for {k} should match"
		
		assert np.allclose(loaded_uvm.freqs, self.uvm.freqs), "Frequencies should match"
		assert loaded_uvm.pols == self.uvm.pols, "Polarizations should match"

		uvm = deepcopy(self.uvm)

		# Invert so that we can also write the inverse coupling
		uvm.invert(first_order=False, multi_path=False)

		# Write the coupling with the inverse
		uvm.write_coupling(file_path, clobber=True)
		loaded_uvm = UVMutualCoupling.read_coupling(file_path)

		# Check if the loaded uvm has the inverse coupling
		assert loaded_uvm.is_inverted, "Loaded uvm should be inverted"
		assert np.allclose(loaded_uvm.inverse_coupling, uvm.inverse_coupling), "Inverse coupling matrices should match"


	def test_validate_data_ok(self):
		# no exceptions on the good cases
		self.uvm._validate_data(self.data)
		self.uvm_time_varying._validate_data(self.data)

	def test_validate_data_ant_mismatch(self):
        # remove one antenna
		bad = self.data
		bad.ants = bad.ants[:-1]
		with pytest.raises(ValueError) as exc:
			self.uvm._validate_data(bad)
		assert "antennas" in str(exc.value)

	def test_validate_data_array_ant_mismatch(self):
		# too few dimensions
		arr = np.zeros((2, 1, 1, self.ntimes, self.nfreqs))
		with pytest.raises(ValueError) as exc:
			self.uvm._validate_data(arr)
		assert "expected shape for coupling" in str(exc.value)

	def test_validate_data_array_pol_mismatch(self):
		# too few dimensions
		arr = np.zeros((1, self.nants, self.nants, self.ntimes, self.nfreqs))
		with pytest.raises(ValueError) as exc:
			self.uvm._validate_data(arr)
		assert "polarizations" in str(exc.value)

	def test_validate_data_freq_mismatch(self):
		# shorten freqs
		bad = self.data
		bad.freqs = bad.freqs[:-1]
		with pytest.raises(ValueError) as exc:
			self.uvm._validate_data(bad)
		assert "frequencies" in str(exc.value)

	def test_validate_data_time_mismatch(self):
		# time-varying uvm expects len(times)==ntimes, so this fails
		bad = self.data
		bad.times = bad.times[:-1]
		with pytest.raises(ValueError) as exc:
			self.uvm_time_varying._validate_data(bad)
		assert "times" in str(exc.value)

	def test_validate_data_missing_pol(self):
		# drop one polarization
		bad = self.data
		bad.pols = bad.pols[:1]
		with pytest.raises(ValueError) as exc:
			self.uvm._validate_data(bad)
		assert "polarization" in str(exc.value)

	def test_validate_data_array_ndim(self):
		# too few dimensions
		arr = np.zeros((2, self.nants, self.nants, self.ntimes))
		with pytest.raises(ValueError) as exc:
			self.uvm._validate_data(arr)
		assert "5 dimensions" in str(exc.value)

	def test_validate_data_array_shape(self):
		# wrong freqs in array branch
		arr = np.zeros((len(self.pols), self.nants, self.nants,
						self.ntimes, self.nfreqs + 1))
		with pytest.raises(ValueError) as exc:
			self.uvm._validate_data(arr)
		assert "does not match expected times and frequencies" in str(exc.value)

class TestValidateShapes:
    @pytest.fixture
    def base_args(self):
        # correct “base” inputs
        nants, nfreqs, ntimes = 3, 4, 2
        pols = ['ee', 'nn']
        antpos = {i: np.array([i, 0, 0]) for i in range(nants)}
        freqs = np.linspace(100e6, 200e6, nfreqs)
        times = np.linspace(2458168.1, 2458168.3, ntimes)
        # correct static and time-varying coupling shapes
        shape_static = (len(pols), nants, nants, 1, nfreqs)
        shape_tv     = (len(pols), nants, nants, ntimes, nfreqs)
        return dict(
            antpos=antpos, freqs=freqs, pols=pols,
            times=times, shape_static=shape_static, shape_tv=shape_tv
        )

    def test_static_coupling_shape_ok(self, base_args):
        # no error for correct static shape and times=None
        c = np.zeros(base_args['shape_static'], dtype=complex)
        UVMutualCoupling(coupling=c, antpos=base_args['antpos'],
                   freqs=base_args['freqs'], pols=base_args['pols'],
                   times=None)

    def test_shape_mismatch_raises(self, base_args):
        # static uvm expects shape_static but we give shape_tv
        c_wrong = np.zeros(base_args['shape_tv'], dtype=complex)
        with pytest.raises(ValueError) as exc:
            UVMutualCoupling(coupling=c_wrong, antpos=base_args['antpos'],
                       freqs=base_args['freqs'], pols=base_args['pols'],
                       times=None)
        assert "times must be provided" in str(exc.value)

    def test_antpos_length_mismatch(self, base_args):
        # drop one antenna from antpos
        antpos_bad = base_args['antpos'].copy()
        antpos_bad.pop(0)
        c = np.zeros(base_args['shape_static'], dtype=complex)
        with pytest.raises(ValueError) as exc:
            UVMutualCoupling(coupling=c, antpos=antpos_bad,
                       freqs=base_args['freqs'], pols=base_args['pols'],
                       times=None)
        assert "antpos length" in str(exc.value)

    def test_freqs_length_mismatch(self, base_args):
        # shorten freqs
        freqs_bad = base_args['freqs'][:-1]
        c = np.zeros(base_args['shape_static'], dtype=complex)
        with pytest.raises(ValueError) as exc:
            UVMutualCoupling(coupling=c, antpos=base_args['antpos'],
                       freqs=freqs_bad, pols=base_args['pols'],
                       times=None)
        assert "freqs length" in str(exc.value)

    def test_times_length_mismatch_for_time_varying(self, base_args):
        # for a time-varying coupling, supply wrong-length times
        c_tv = np.zeros(base_args['shape_tv'], dtype=complex)
        times_bad = base_args['times'][:-1]
        with pytest.raises(ValueError) as exc:
            UVMutualCoupling(coupling=c_tv, antpos=base_args['antpos'],
                       freqs=base_args['freqs'], pols=base_args['pols'],
                       times=times_bad)
        assert "times length" in str(exc.value)