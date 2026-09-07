import os,unittest
from iris_models import np,load_data,split_data,fit_preprocessor,preprocess,forward,initialize
import cudaq

class IrisTests(unittest.TestCase):
    def test_source_and_disjoint_groups(self):
        rows,_=load_data();self.assertGreaterEqual(len(rows),95)
        for seed in (2028,2029):
            splits=split_data(rows,seed);seen=set()
            for split in splits.values():
                keys={tuple(r['features']) for r in split};self.assertFalse(seen&keys);seen|=keys
            self.assertEqual(sum(map(len,splits.values())),len(rows))

    def test_pca_fit_and_no_refit(self):
        rows,_=load_data();splits=split_data(rows,2028);raw=[r['features'] for r in splits['train']];p=fit_preprocessor(raw)
        np.testing.assert_allclose(p['mean'],np.mean(raw,axis=0));components=np.array(p['components']);np.testing.assert_allclose(components@components.T,np.eye(2),atol=1e-12)
        x,flags=preprocess(raw,p);self.assertEqual(x.shape,(len(raw),2));self.assertFalse(flags.any())
        before=str(p);preprocess([r['features'] for r in splits['test']],p);self.assertEqual(str(p),before)

    def test_models_against_cudaq(self):
        cudaq.set_target(os.environ.get('DAY20_TARGET','qpp-cpu'))
        for model in ('mlp','vqc','hybrid'):
            w=initialize(model,42);x=[[.2,-.4],[.7,.3]]
            np.testing.assert_allclose(forward(model,x,w,'cudaq'),forward(model,x,w,'numpy'),atol=1e-5)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):fit_preprocessor([[0]*4]*3)
        with self.assertRaises(ValueError):forward('mlp',[[0,0]],[0]*8)

if __name__=='__main__':unittest.main()
