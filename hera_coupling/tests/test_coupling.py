import pytest
import numpy as np
from copy import deepcopy

# Import hera modules
import hera_coupling as hm
from hera_coupling import UVCoupling
from hera_coupling.coupling import _extract_data_matrix
from hera_cal.datacontainer import DataContainer

try:
	import symengine as sympy
	import_sympy = True
except ImportError:
	import_sympy = False


def test_apply_coupling_sympy():
	# not ready yet
	if True:
		return

	if not import_sympy:
		return

	# can only use 1 freq and time due to sympy constraints
	freqs = np.linspace(120e6, 180e6, 1)
	times = np.linspace(2458168.1, 2458168.3, 1)

	#ants = ...

	# create redundant visibility matrix in sympy
	Vr = []
	Vdata = {}
	for i in range(len(ants)):
	    _V = []
	    for j in range(len(ants)):
	        bl = (i, j)
	        if j >= i:
	            _V.append("V_{}".format(bl2red_idx[bl]))
	            Vdata["V_{}".format(bl2red_idx[bl])] = vd[bl2red[bl]].item()
	        else:
	            _V.append(sympy.conjugate("V_{}".format(bl2red_idx[bl[::-1]])))
	    Vr.append(_V)
	Vr = sympy.Matrix(Vr)

	# create redundant coupling matrix in sympy
	Er = []
	red_vecs = torch.zeros(0, 3)
	red_num = 0
	for i in range(len(ants)):
		_E = []
		for j in range(len(ants)):
			bl_vec = rvis_cpl.antpos[j] - rvis_cpl.antpos[i]
			diff_norm = (red_vecs - bl_vec).norm(dim=-1).isclose(torch.tensor(0.), atol=1.0)
			if diff_norm.any():
				# found a match in red_vecs
				_E.append("e_{}".format(diff_norm.argwhere()[0,0]))
			else:
				# no match in red_vecs, new red_bl
				red_vecs = torch.cat([red_vecs, bl_vec[None, :]], dim=0)
				_E.append("e_{}".format(red_num))
				red_num += 1
		Er.append(_E)

	for i in range(len(ants)):        
		Er[i][i] = '1 + {}'.format(Er[i][i])

	X = sympy.Matrix(Er)

	# perform coupling operation
	Vc = Er @ Vr @ Er.conjugate().T

	# substitute in data and params
	Vc = np.array(Vc.subs(dict(
	    list(Vdata.items()) + \
	    list({f'e_{i}': rvis_cpl.params[0,0,i,0,0].item() for i in range(len(rvis_cpl.coupling_terms))}.items())
		))).astype(np.complex128)


	# apply coupling to data
	#vout = ... 

	# compare against analytic result
	#r = vout[[bl for bl in vout.bls]] / np.array([Vc[bl[0], bl[1]] for bl in vout.bls])
	assert np.isclose(r, 1 + 0j, atol=1e-10).all()

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
		self.uvc = UVCoupling(
			coupling=coupling,
			antpos=self.antpos,
			freqs=self.freqs,
			pols=self.pols,
		)

		shape = (2, self.nants, self.nants, self.ntimes, self.nfreqs)
		coupling = rng.normal(0, 1e-1, shape) + 0j
		coupling = (np.swapaxes(coupling, 1, 2).conj() + coupling) / 2
		self.uvc_time_varying = UVCoupling(
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

		# Invert the coupling
		coupled_data = self.uvc.apply(
			data, 
			forward=True,
			first_order=first_order,
			multi_path=multi_path,
			inplace=False
		)

		uncoupled_data = self.uvc.apply(
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

		coupled_data = self.uvc_time_varying.apply(
			data, 
			forward=True,
			first_order=first_order,
			multi_path=multi_path,
			inplace=False
		)

		uncoupled_data = self.uvc_time_varying.apply(
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
		uvc = deepcopy(self.uvc)
		uvc_time_varying = deepcopy(self.uvc_time_varying)
		
		# Invert the coupling
		uvc.invert(first_order=first_order, multi_path=multi_path)
		uvc_time_varying.invert(first_order=first_order, multi_path=multi_path)

		if not first_order:
			# Check if the coupling matrix is Hermitian
			assert uvc.is_inverted, "Coupling should be inverted"
			assert uvc.inverse_coupling.shape == uvc.coupling.shape, "Inverse coupling shape mismatch" 
			assert uvc_time_varying.is_inverted, "Coupling should be inverted"
			assert uvc_time_varying.inverse_coupling.shape == uvc_time_varying.coupling.shape, "Inverse coupling shape mismatch" 
		else:
			# For first order, Sylvester solver is used, so 
			assert not uvc.is_inverted, "First order coupling should not be inverted"
			assert not hasattr(uvc, "inverse_coupling"), "First order coupling should not have inverse_coupling attribute"
			assert not uvc_time_varying.is_inverted, "First order coupling should not be inverted"
			assert not hasattr(uvc_time_varying, "inverse_coupling"), "First order coupling should not have inverse_coupling attribute"

	def test_coupling_io(self, tmp_path):
		# Test saving and loading the coupling
		file_path = tmp_path / "test_coupling.hdf5"
		self.uvc.write_coupling(file_path)
	
		with pytest.raises(FileExistsError):
			# Attempt to overwrite the file without overwrite flag
			self.uvc.write_coupling(file_path, clobber=False)

		loaded_uvc = UVCoupling.read_coupling(file_path)
		with pytest.raises(FileNotFoundError):
			# Attempt to read a non-existent file
			UVCoupling.read_coupling(tmp_path / "non_existent_file.hdf5")
		
		assert isinstance(loaded_uvc, UVCoupling), "Loaded object should be a UVCoupling"
		assert np.allclose(loaded_uvc.coupling, self.uvc.coupling), "Coupling matrices should match"
		for k in self.uvc.antpos:
			assert np.allclose(loaded_uvc.antpos[k], self.uvc.antpos[k]), f"Antenna position for {k} should match"
		
		assert np.allclose(loaded_uvc.freqs, self.uvc.freqs), "Frequencies should match"
		assert loaded_uvc.pols == self.uvc.pols, "Polarizations should match"

	def test_validate_data_ok(self):
		# no exceptions on the good cases
		self.uvc._validate_data(self.data)
		self.uvc_time_varying._validate_data(self.data)

	def test_validate_data_ant_mismatch(self):
        # remove one antenna
		bad = self.data
		bad.ants = bad.ants[:-1]
		with pytest.raises(ValueError) as exc:
			self.uvc._validate_data(bad)
		assert "antennas" in str(exc.value)

	def test_validate_data_array_ant_mismatch(self):
		# too few dimensions
		arr = np.zeros((2, 1, 1, self.ntimes, self.nfreqs))
		with pytest.raises(ValueError) as exc:
			self.uvc._validate_data(arr)
		assert "expected shape for coupling" in str(exc.value)

	def test_validate_data_array_pol_mismatch(self):
		# too few dimensions
		arr = np.zeros((1, self.nants, self.nants, self.ntimes, self.nfreqs))
		with pytest.raises(ValueError) as exc:
			self.uvc._validate_data(arr)
		assert "polarizations" in str(exc.value)

	def test_validate_data_freq_mismatch(self):
		# shorten freqs
		bad = self.data
		bad.freqs = bad.freqs[:-1]
		with pytest.raises(ValueError) as exc:
			self.uvc._validate_data(bad)
		assert "frequencies" in str(exc.value)

	def test_validate_data_time_mismatch(self):
		# time-varying uvc expects len(times)==ntimes, so this fails
		bad = self.data
		bad.times = bad.times[:-1]
		with pytest.raises(ValueError) as exc:
			self.uvc_time_varying._validate_data(bad)
		assert "times" in str(exc.value)

	def test_validate_data_missing_pol(self):
		# drop one polarization
		bad = self.data
		bad.pols = bad.pols[:1]
		with pytest.raises(ValueError) as exc:
			self.uvc._validate_data(bad)
		assert "polarization" in str(exc.value)

	def test_validate_data_array_ndim(self):
		# too few dimensions
		arr = np.zeros((2, self.nants, self.nants, self.ntimes))
		with pytest.raises(ValueError) as exc:
			self.uvc._validate_data(arr)
		assert "5 dimensions" in str(exc.value)

	def test_validate_data_array_shape(self):
		# wrong freqs in array branch
		arr = np.zeros((len(self.pols), self.nants, self.nants,
						self.ntimes, self.nfreqs + 1))
		with pytest.raises(ValueError) as exc:
			self.uvc._validate_data(arr)
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
        UVCoupling(coupling=c, antpos=base_args['antpos'],
                   freqs=base_args['freqs'], pols=base_args['pols'],
                   times=None)

    def test_shape_mismatch_raises(self, base_args):
        # static UVCoupling expects shape_static but we give shape_tv
        c_wrong = np.zeros(base_args['shape_tv'], dtype=complex)
        with pytest.raises(ValueError) as exc:
            UVCoupling(coupling=c_wrong, antpos=base_args['antpos'],
                       freqs=base_args['freqs'], pols=base_args['pols'],
                       times=None)
        assert "times must be provided" in str(exc.value)

    def test_antpos_length_mismatch(self, base_args):
        # drop one antenna from antpos
        antpos_bad = base_args['antpos'].copy()
        antpos_bad.pop(0)
        c = np.zeros(base_args['shape_static'], dtype=complex)
        with pytest.raises(ValueError) as exc:
            UVCoupling(coupling=c, antpos=antpos_bad,
                       freqs=base_args['freqs'], pols=base_args['pols'],
                       times=None)
        assert "antpos length" in str(exc.value)

    def test_freqs_length_mismatch(self, base_args):
        # shorten freqs
        freqs_bad = base_args['freqs'][:-1]
        c = np.zeros(base_args['shape_static'], dtype=complex)
        with pytest.raises(ValueError) as exc:
            UVCoupling(coupling=c, antpos=base_args['antpos'],
                       freqs=freqs_bad, pols=base_args['pols'],
                       times=None)
        assert "freqs length" in str(exc.value)

    def test_times_length_mismatch_for_time_varying(self, base_args):
        # for a time-varying coupling, supply wrong-length times
        c_tv = np.zeros(base_args['shape_tv'], dtype=complex)
        times_bad = base_args['times'][:-1]
        with pytest.raises(ValueError) as exc:
            UVCoupling(coupling=c_tv, antpos=base_args['antpos'],
                       freqs=base_args['freqs'], pols=base_args['pols'],
                       times=times_bad)
        assert "times length" in str(exc.value)