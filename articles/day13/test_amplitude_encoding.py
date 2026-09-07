import os
import unittest
from amplitude_encoding import cudaq,np,normalize,gate_angles,gates,loaded,simulator_state,extract

class AmplitudeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cudaq.set_target(os.environ.get('DAY13_TARGET','qpp-cpu'))

    def test_normalization_padding_no_mutation(self):
        v=np.array([3.,4.,0.]);copy=v.copy();a,m=normalize(v)
        np.testing.assert_allclose(a,[.6,.8,0,0]);np.testing.assert_array_equal(v,copy)
        self.assertEqual(m['qubits'],2)
        self.assertEqual(m['norm_scale']*m['norm_scaled'],5.)

    def test_invalid_and_extreme_values(self):
        for v in ([],[1],[0,0],[1,np.nan],[1,np.inf],[1j,1],[[1,2]],[1]*5):
            with self.assertRaises(ValueError):normalize(v)
        for v in ([1e308,-1e308],[1e-308,1e-308]):
            a,_=normalize(v);self.assertAlmostEqual(float(a@a),1.)
        with self.assertRaises(ValueError):gate_angles([1,1])

    def test_scale_and_global_sign(self):
        a,_=normalize([3,4]);b,_=normalize([30,40]);c,_=normalize([-3,-4])
        np.testing.assert_allclose(a,b);self.assertAlmostEqual(abs(np.vdot(a,c))**2,1.)

    def test_relative_sign_probabilities(self):
        a,_=normalize([3,4]);b,_=normalize([3,-4])
        np.testing.assert_allclose(a*a,b*b)
        self.assertAlmostEqual(abs(np.vdot(a,b))**2,.0784)

    def test_all_basis_states_loading_order(self):
        for i in range(4):
            a=np.eye(4)[i];angles=gate_angles(a)
            np.testing.assert_allclose(abs(extract(cudaq.get_state(gates,angles,2),2))**2,a,atol=1e-5)
            np.testing.assert_allclose(abs(extract(cudaq.get_state(loaded,simulator_state(a)),2))**2,a,atol=1e-5)

    def test_signed_random_vectors_and_zero_branches(self):
        rng=np.random.default_rng(42)
        vectors=[rng.normal(size=d) for d in (2,3,4) for _ in range(3)]
        vectors += [[0,0,-3,4],[-3,4,0,0]]
        for v in vectors:
            a,m=normalize(v);n=m['qubits']
            actual=extract(cudaq.get_state(gates,gate_angles(a),n),n)
            self.assertAlmostEqual(abs(np.vdot(a,actual))**2,1.,delta=1e-5)

if __name__=='__main__':unittest.main()
