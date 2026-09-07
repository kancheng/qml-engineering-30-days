import unittest
import os
from benchmark import np,cudaq,forward,search,dataset,fit_scaler,transform

class BenchmarkTests(unittest.TestCase):
    def test_exact_budget_and_monotonicity(self):
        for count in (3,4,9):
            initial=np.ones(count);w,h,s=search(lambda w:float(w@w),initial,109)
            self.assertEqual(h[-1]['evaluations'],109)
            self.assertTrue(all(b['loss']<=a['loss'] for a,b in zip(h,h[1:])))
            np.testing.assert_array_equal(initial,np.ones(count))
        with self.assertRaises(ValueError):search(lambda w:0.,[0],4)

    def test_logistic_formula(self):
        np.testing.assert_allclose(forward('logistic',[[0,0],[1,-1]],[1,2,.5]),1/(1+np.exp(-np.array([.5,-.5]))))
        with self.assertRaises(ValueError):forward('logistic',[[0,0]],[0]*4)

    def test_dataset_and_scaler_boundary(self):
        data=dataset(2027);seen=set()
        for split in data.values():
            points=set(map(tuple,split['raw']));self.assertFalse(seen&points);seen|=points
        scaler=fit_scaler(data['train']['raw']);before=dict(scaler)
        transform(data['test']['raw'],scaler);self.assertEqual(scaler,before)

    def test_vqc_reference(self):
        cudaq.set_target(os.environ.get('DAY18_TARGET','qpp-cpu'))
        x=[[.2,-.4],[.7,.3]];w=np.random.default_rng(42).normal(0,.2,4)
        np.testing.assert_allclose(forward('vqc',x,w),forward('vqc',x,w,True),atol=1e-5)

if __name__=='__main__':unittest.main()
