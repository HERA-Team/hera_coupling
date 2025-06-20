import numpy as np

import hera_coupling as hm

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


