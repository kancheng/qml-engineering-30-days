import unittest
import os
from encoding import np, cudaq, fit_scaler, transform, angle_reference, basis_reference, amplitude_reference, encoded

class EncodingTests(unittest.TestCase):
    def test_train_only_scaling_and_no_mutation(self):
        scaler=fit_scaler([[10,100],[40,200]])
        raw=np.array([[25.,150.],[5.,250.]])
        copy=raw.copy()
        scaled,mask=transform(raw,scaler)
        np.testing.assert_allclose(scaled,[[0,0],[-1,1]])
        np.testing.assert_array_equal(mask,[[False,False],[True,True]])
        np.testing.assert_array_equal(raw,copy)
        self.assertEqual(scaler['minimum'],[10,100])

    def test_invalid_inputs(self):
        for data in ([],[[1,2]],[[0,0],[1,float('nan')]],[[1,2,3]]):
            with self.assertRaises(ValueError): fit_scaler(data)
        with self.assertRaises(ValueError): amplitude_reference([0,0])
        with self.assertRaises(ValueError): basis_reference([0,0.5])
        with self.assertRaises(ValueError): angle_reference([2,0])

    def test_basis_and_amplitude_semantics(self):
        np.testing.assert_array_equal(basis_reference([1,0]),[0,0,1,0])
        np.testing.assert_allclose(amplitude_reference([3,4,0]),[.6,.8,0,0])
        np.testing.assert_allclose(amplitude_reference([3,4]),amplitude_reference([30,40]))
        self.assertAlmostEqual(np.linalg.norm(amplitude_reference([1e308,1e308])),1.)

    def test_angle_endpoints_collide(self):
        a,b=angle_reference([-1,.2]),angle_reference([1,.2])
        self.assertAlmostEqual(abs(np.vdot(a,b))**2,1.)

    def test_z_probabilities_hide_sign(self):
        a,b=angle_reference([.25,0]),angle_reference([-.25,0])
        np.testing.assert_allclose(a*a,b*b)
        self.assertLess(abs(np.vdot(a,b))**2,1.)

    def test_cudaq_basis_order_and_x_readout(self):
        cudaq.set_target(os.environ.get('DAY11_TARGET','qpp-cpu'))
        for feature in ([1,0],[0,1],[.25,-.3],[-.25,-.3]):
            angles=(np.pi*np.array(feature)).tolist()
            state=cudaq.get_state(encoded,angles)
            actual=np.array([state.amplitude(k) for k in ['00','01','10','11']])
            self.assertAlmostEqual(abs(np.vdot(angle_reference(feature),actual))**2,1.,delta=1e-5)
            x=cudaq.observe(encoded,cudaq.spin.x(0),angles,shots_count=-1).expectation()
            self.assertAlmostEqual(x,np.sin(angles[0]),delta=1e-5)

if __name__=='__main__': unittest.main()
