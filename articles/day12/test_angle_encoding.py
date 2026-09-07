import os
import unittest
from angle_encoding import (np, cudaq, MAPS, AXES, to_angles, reference, state_vector,
                            analytic_bloch, product_fidelity, X, Y, Z, I)

class AngleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cudaq.set_target(os.environ.get('DAY12_TARGET','qpp-cpu'))

    def test_mapping_bounds_and_validation(self):
        for mode, expected in [('centered_full',[-np.pi,np.pi]),('centered_half',[-np.pi/2,np.pi/2]),('positive_half',[0,np.pi])]:
            np.testing.assert_allclose(to_angles([-1,1],mode),expected)
        for data in ([0],[0,float('nan')],[0,2]):
            with self.assertRaises(ValueError): to_angles(data)
        with self.assertRaises(ValueError): to_angles([0,0],'unknown')

    def test_day11_compatibility(self):
        from articles.day11.encoding import angle_reference
        np.testing.assert_allclose(reference(to_angles([.2,-.7],'centered_full')),angle_reference([.2,-.7]))

    def test_endpoint_and_pair_overlap(self):
        for mode in MAPS:
            a,b=to_angles([-1,.2],mode),to_angles([1,.2],mode)
            expected=1. if mode=='centered_full' else 0.
            self.assertAlmostEqual(product_fidelity(a,b),expected)
            self.assertAlmostEqual(abs(np.vdot(state_vector(a),state_vector(b)))**2,expected,delta=1e-5)

    def test_bloch_signs_against_matrices(self):
        for axis in AXES:
            state=reference([.7,-.3],axis)
            actual=[float(np.vdot(state,np.kron(op,I)@state).real) for op in (X,Y,Z)]
            np.testing.assert_allclose(actual,analytic_bloch(.7,axis),atol=1e-12)

    def test_cudaq_states_all_axes_and_complex_phases(self):
        for axis in AXES:
            a=[.7,-1.1]
            self.assertAlmostEqual(abs(np.vdot(reference(a,axis),state_vector(a,axis)))**2,1.,delta=1e-5)

    def test_rz_invisibility_and_preparation_order(self):
        a,b=[.2,.4],[1.7,-.8]
        self.assertAlmostEqual(abs(np.vdot(state_vector(a,'rz'),state_vector(b,'rz')))**2,1.,delta=1e-5)
        self.assertLess(abs(np.vdot(state_vector(a,'h_rz'),state_vector(b,'h_rz')))**2,.9)

    def test_positive_half_z_is_monotone(self):
        values=[analytic_bloch(to_angles([x,0])[0],'ry')[2] for x in np.linspace(-1,1,9)]
        self.assertTrue(all(a>b for a,b in zip(values,values[1:])))

if __name__=='__main__': unittest.main()
