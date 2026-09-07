import copy, os, unittest
import cudaq
from kernels import np, kernel, state, solve, fit_scaler, transform, load_data, split_data
from articles.day04.bell_state import CNOT

class KernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cudaq.set_target(os.getenv('DAY23_TARGET','qpp-cpu'))
        cls.x=np.array([[-1,-1,-1,-1],[1,-1,-1,-1],[.2,-.4,.6,-.8],[.7,.1,-.3,.5]])

    def test_compute_uncompute(self):
        expected=kernel('quantum',self.x,self.x)
        actual=kernel('quantum',self.x,self.x,'cudaq')
        np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=0)
        self.assertAlmostEqual(expected[0,1],0,places=12)
        np.testing.assert_allclose(np.diag(actual),1,atol=1e-5)

    def test_psd_and_density_feature_map(self):
        states=np.array([state(x) for x in self.x])
        rho=np.array([np.outer(s,s.conj()).reshape(-1) for s in states])
        np.testing.assert_allclose(kernel('quantum',self.x,self.x), (rho.conj()@rho.T).real,atol=1e-12)
        for name in ('quantum','rbf','linear'):
            k=kernel(name,self.x,self.x)
            np.testing.assert_allclose(k,k.T,atol=1e-12)
            self.assertGreaterEqual(np.linalg.eigvalsh(k).min(),-1e-12)

    def test_common_final_unitary_cancels(self):
        a,b=state(self.x[2]),state(self.x[3])
        self.assertAlmostEqual(abs(np.vdot(a,b))**2,abs(np.vdot(CNOT@a,CNOT@b))**2,places=12)

    def test_ridge_primal_dual_and_order(self):
        phi=np.column_stack([np.ones(len(self.x)),self.x/2]); y=np.array([0,0,1,1]); lam=.1
        k=kernel('linear',self.x,self.x); alpha=solve(k,y,lam)
        beta=np.linalg.solve(phi.T@phi+lam*np.eye(5),phi.T@y)
        np.testing.assert_allclose(k@alpha,phi@beta,atol=1e-12)
        order=[2,0,3,1]
        np.testing.assert_allclose(k[:,order]@solve(k[np.ix_(order,order)],y[order],lam),k@alpha,atol=1e-12)

    def test_train_only(self):
        rows,_=load_data();splits=split_data(rows,2028)
        train=[r['features'] for r in splits['train']];prep=fit_scaler(train);before=copy.deepcopy(prep)
        _,flags=transform(np.array(train)+100,prep)
        self.assertTrue(flags.all());self.assertEqual(before,prep)
        np.testing.assert_allclose(prep['minimum'],np.min(train,axis=0))

    def test_invalid(self):
        with self.assertRaises(ValueError): kernel('quantum',[[2]*4],self.x)
        with self.assertRaises(ValueError): solve(np.eye(2),[0,1],0)
        with self.assertRaises(ValueError): fit_scaler(np.ones((3,4)))
        with self.assertRaises(ValueError): kernel('missing',self.x,self.x)

if __name__=='__main__': unittest.main()
