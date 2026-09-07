import os
import unittest
from models import np,cudaq,initialize,mlp_initialize,mlp_forward,qnn_probability

class ModelComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cudaq.set_target(os.environ.get('DAY16_TARGET','qpp-cpu'))

    def test_mlp_zero_and_bias(self):
        x=[[.2,-.4],[1,1]];w=np.zeros(9)
        np.testing.assert_allclose(mlp_forward(x,w),[.5,.5])
        w[8]=2.;np.testing.assert_allclose(mlp_forward(x,w),[1/(1+np.exp(-2))]*2)

    def test_mlp_scalar_reference_and_shapes(self):
        w=mlp_initialize();x=[.25,-.4]
        h0=np.tanh(x[0]*w[0]+x[1]*w[2]+w[4]);h1=np.tanh(x[0]*w[1]+x[1]*w[3]+w[5])
        expected=1/(1+np.exp(-(h0*w[6]+h1*w[7]+w[8])))
        self.assertAlmostEqual(float(mlp_forward([x],w)[0]),expected)
        for features,weights in [([],w),([[1]],w),([[0,np.nan]],w),([x],[0]*8)]:
            with self.assertRaises(ValueError):mlp_forward(features,weights)

    def test_initialization_and_nonmutation(self):
        w=mlp_initialize();copy=w.copy();x=np.array([[.2,-.1]]);xc=x.copy()
        mlp_forward(x,w);np.testing.assert_array_equal(w,copy);np.testing.assert_array_equal(x,xc)
        np.testing.assert_array_equal(w,mlp_initialize())
        self.assertEqual(len(initialize()),4);self.assertEqual(len(w),9)

    def test_qnn_periodic_weights(self):
        w=initialize();x=[.25,-.4];value=qnn_probability(x,w)
        for i in range(4):
            changed=w.copy();changed[i]+=2*np.pi
            self.assertAlmostEqual(qnn_probability(x,changed),value,delta=1e-5)

    def test_cudaq_reference_and_parameter_response(self):
        w=initialize();x=[.25,-.4]
        self.assertAlmostEqual(qnn_probability(x,w),qnn_probability(x,w,True),delta=1e-5)
        changed=w.copy();changed[1]+=.5
        self.assertGreater(abs(qnn_probability(x,w)-qnn_probability(x,changed)),1e-3)

    def test_state_linearity_and_nonlinear_angle_readout(self):
        from articles.day03.quantum_gates import ry,KET_ZERO,KET_ONE
        u=ry(.7);a=.3;b=-.4
        np.testing.assert_allclose(u@(a*KET_ZERO+b*KET_ONE),a*(u@KET_ZERO)+b*(u@KET_ONE))
        self.assertGreater(abs(np.cos(np.pi/4)-.5*(1+np.cos(np.pi/2))),.1)

if __name__=='__main__':unittest.main()
