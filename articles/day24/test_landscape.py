import os,unittest
import cudaq
from landscape import np,initialize,reference,shift,validate

class LandscapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cudaq.set_target(os.getenv('DAY24_TARGET','qpp-cpu'))

    def test_depth1_analytic(self):
        for n in (2,4,8):
            w=initialize(n,1,'uniform',42);cost,g=reference(n,1,w);angles=w[::2]
            self.assertAlmostEqual(cost['local'],np.cos(angles[0]),places=12)
            self.assertAlmostEqual(cost['global'],np.prod(np.cos(angles)),places=12)
            self.assertAlmostEqual(g['local'],-np.sin(angles[0]),places=12)
            self.assertAlmostEqual(g['global'],-np.sin(angles[0])*np.prod(np.cos(angles[1:])),places=12)

    def test_shift_targets(self):
        for n,depth in ((2,1),(4,4),(8,8)):
            for mode in ('uniform','small'):
                w=initialize(n,depth,mode,42);_,g=reference(n,depth,w)
                for cost in ('local','global'):
                    actual,_=shift(n,depth,w,cost);self.assertAlmostEqual(actual,g[cost],delta=1e-5)

    def test_finite_difference(self):
        w=initialize(4,4,'uniform',17);_,g=reference(4,4,w);a=w.copy();b=w.copy();a[0]+=1e-5;b[0]-=1e-5
        for cost in ('local','global'):
            fd=(reference(4,4,a)[0][cost]-reference(4,4,b)[0][cost])/2e-5
            self.assertAlmostEqual(fd,g[cost],delta=1e-8)

    def test_zero_initialization_stationary(self):
        costs,g=reference(4,4,np.zeros(32))
        for cost in ('local','global'):
            self.assertAlmostEqual(costs[cost],1)
            self.assertAlmostEqual(g[cost],0)

    def test_invalid_and_reproducible(self):
        np.testing.assert_array_equal(initialize(4,4,'small',42),initialize(4,4,'small',42))
        with self.assertRaises(ValueError):validate(1,1,[])
        with self.assertRaises(ValueError):validate(4,4,np.zeros(31))
        with self.assertRaises(ValueError):initialize(4,4,'missing',42)

if __name__=='__main__':unittest.main()
