import os
import unittest
from gradients import np,Config,predict,parameter_shift,matrix_gradient,difference,loss_and_gradient
from articles.day14.model import cudaq,initialize,reference_prediction

class GradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cudaq.set_target(os.environ.get('DAY17_TARGET','qpp-cpu'))

    def test_difference_and_validation(self):
        np.testing.assert_allclose(difference(lambda w:float(w@w),[2.,-3.],1e-4),[4,-6])
        for step in (0,-1,np.nan):
            with self.assertRaises(ValueError):difference(lambda w:0.,[1.],step)
        with self.assertRaises(ValueError):loss_and_gradient([],[],initialize())

    def test_shift_matrix_and_finite_difference(self):
        for encoding,x in [('angle',[.25,-.4]),('amplitude',[1,-2,3,-4])]:
            config=Config(encoding,2);w=initialize(2);copy=w.copy()
            expected=matrix_gradient(x,w,config)
            np.testing.assert_allclose(parameter_shift(x,w,config),expected,atol=1e-5)
            np.testing.assert_allclose(difference(lambda v:reference_prediction(x,v,config),w,1e-5),expected,atol=1e-8)
            np.testing.assert_array_equal(w,copy)

    def test_chain_rule(self):
        x=[[.25,-.4],[-.6,.2]];y=[1,0];w=initialize()
        loss,g=loss_and_gradient(x,y,w)
        ref=difference(lambda v:np.mean([((1-reference_prediction(a,v))/2-b)**2 for a,b in zip(x,y)]),w,1e-5)
        np.testing.assert_allclose(g,ref,atol=1e-5)

    def test_loss_shift_is_not_general_rule(self):
        theta=.4;f=lambda a:(1-np.cos(a))/2
        true=2*(f(theta)-1)*np.sin(theta)/2
        wrong=((f(theta+np.pi/2)-1)**2-(f(theta-np.pi/2)-1)**2)/2
        self.assertGreater(abs(true-wrong),.01)

    def test_shared_parameter_counterexample(self):
        t=.3;f=lambda a:np.cos(2*a)
        wrong=(f(t+np.pi/2)-f(t-np.pi/2))/2
        self.assertAlmostEqual(wrong,0.)
        self.assertGreater(abs(-2*np.sin(2*t)-wrong),1.)

if __name__=='__main__':unittest.main()
