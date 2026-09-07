import os,copy,unittest
import cudaq
from wine_models import np,load_data,split_data,fit,representation,forward,COUNTS

class WineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cudaq.set_target(os.getenv('DAY25_TARGET','qpp-cpu'))
        cls.rows,cls.source=load_data();cls.splits=split_data(cls.rows,2030)
        cls.x=np.array([r['features'] for r in cls.splits['train']]);cls.y=[r['label'] for r in cls.splits['train']]
        cls.prep=fit(cls.x,cls.y)

    def test_source_and_split(self):
        self.assertEqual(len(self.rows),119);self.assertEqual(self.source['dropped_duplicate_ids'],[])
        self.assertEqual(sum(r['label']==0 for r in self.rows),71)
        self.assertEqual([len(v) for v in self.splits.values()],[70,23,26])
        sets=[{tuple(r['features']) for r in rows} for rows in self.splits.values()]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])

    def test_train_only_pca(self):
        np.testing.assert_allclose(self.prep['mean'],self.x.mean(0))
        c=np.array(self.prep['components']);np.testing.assert_allclose(c@c.T,np.eye(2),atol=1e-12)
        self.assertEqual(self.prep,fit(self.x,np.zeros(len(self.x))))
        before=copy.deepcopy(self.prep)
        for model,count in COUNTS.items():representation(model,self.x+100,self.prep,np.zeros(count))
        self.assertEqual(before,self.prep)
        _,flags=representation('vqc',self.x,self.prep,np.zeros(4));self.assertFalse(flags.any())

    def test_all_models_reference(self):
        for model,count in COUNTS.items():
            w=np.random.default_rng(42).normal(0,.2,count)
            if model=='hybrid':w[10]=.8
            np.testing.assert_allclose(forward(model,self.x[:3],self.prep,w,'cudaq'),
                                       forward(model,self.x[:3],self.prep,w),atol=1e-5,rtol=0)

    def test_full_features_and_shared_projection(self):
        w=np.zeros(14);w[12]=1
        z=(self.x-self.prep['mean'])/self.prep['std']
        np.testing.assert_allclose(forward('logistic13',self.x,self.prep,w),.5*(1+np.tanh(z[:,12]/2)))
        projections=[representation(m,self.x,self.prep,np.zeros(COUNTS[m]))[0] for m in ('mlp','vqc','hybrid')]
        np.testing.assert_array_equal(projections[0],projections[1]);np.testing.assert_array_equal(projections[0],projections[2])

    def test_invalid(self):
        with self.assertRaises(ValueError):fit(np.ones((3,13)),[0,1,1])
        with self.assertRaises(ValueError):forward('vqc',self.x[:,:4],self.prep,np.zeros(4))
        with self.assertRaises(ValueError):forward('logistic13',self.x,self.prep,np.zeros(13))

if __name__=='__main__':unittest.main()
