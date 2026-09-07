import os,unittest
from hybrid import np,cudaq,initialize,forward,loss_and_gradient,quantum_jacobian,quantum_value
from articles.day17.gradients import difference

class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cudaq.set_target(os.environ.get('DAY19_TARGET','qpp-cpu'))

    def test_forward_reference_and_nonmutation(self):
        x=np.array([[.2,-.4],[.7,.3]]);v=initialize();xc=x.copy();vc=v.copy()
        np.testing.assert_allclose(forward(x,v)[0],forward(x,v,True)[0],atol=1e-5)
        np.testing.assert_array_equal(x,xc);np.testing.assert_array_equal(v,vc)
        self.assertTrue(np.all(abs(forward(x,v)[1]['angles'])<=np.pi))

    def test_full_chain_rule_all_groups(self):
        x=[[.2,-.4],[.7,.3]];y=[1,0];v=initialize()
        loss,g=loss_and_gradient(x,y,v)
        expected=difference(lambda w:np.mean((forward(x,w,True)[0]-y)**2),v,1e-5)
        np.testing.assert_allclose(g,expected,atol=1e-5)
        for group in (g[:6],g[6:10],g[10:]):self.assertGreater(np.linalg.norm(group),1e-6)

    def test_quantum_input_and_weight_jacobian(self):
        angles=[.4,-.7];w=initialize()[6:10]
        expected=difference(lambda v:quantum_value(v[:2],v[2:],True),np.r_[angles,w],1e-5)
        np.testing.assert_allclose(quantum_jacobian(angles,w),expected,atol=1e-5)

    def test_zero_head_scale_blocks_upstream_gradient(self):
        v=initialize();v[10]=0
        _,g=loss_and_gradient([[.2,-.4]],[1],v)
        np.testing.assert_allclose(g[:10],0,atol=1e-12)
        self.assertGreater(abs(g[11]),0)

    def test_invalid_contracts(self):
        for x,v in [([],initialize()),([[1]],initialize()),([[0,np.nan]],initialize()),([[0,0]],[0]*11)]:
            with self.assertRaises(ValueError):forward(x,v)
        with self.assertRaises(ValueError):loss_and_gradient([[0,0]],[2],initialize())

if __name__=='__main__':unittest.main()
