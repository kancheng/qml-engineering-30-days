import os
import tempfile
import json
import unittest
from pathlib import Path
from classifier import np,dataset,probabilities,metrics,fit_scaler,predict_proba,load_checkpoint
from articles.day14.model import cudaq,initialize

class ClassifierTests(unittest.TestCase):
    def test_split_labels_and_reproducibility(self):
        data=dataset();self.assertEqual(data,dataset());seen=set()
        for split in data.values():
            x=np.array(split['raw']);y=np.array(split['labels'])
            self.assertEqual(sum(y),6)
            np.testing.assert_array_equal(y,(x[:,0]*x[:,1]<0).astype(int))
            points=set(map(tuple,x));self.assertFalse(seen&points);seen|=points
        self.assertEqual(len(seen),36)

    def test_probability_and_threshold(self):
        np.testing.assert_allclose(probabilities([-1,0,1]),[1,.5,0])
        result=metrics([0,1,1],[.2,.5,.1])
        self.assertEqual(result['confusion_matrix'],[[1,0],[1,1]])
        self.assertAlmostEqual(result['accuracy'],2/3)
        with self.assertRaises(ValueError):probabilities([1.1])
        with self.assertRaises(ValueError):metrics([2],[.5])

    def test_probability_roundoff_only(self):
        np.testing.assert_allclose(probabilities([1+1e-7,-1-1e-7]),[0,1])
        with self.assertRaises(ValueError):probabilities([np.nan])
        with self.assertRaises(ValueError):metrics([],[])

    def test_checkpoint_roundtrip_and_forward(self):
        cudaq.set_target(os.environ.get('DAY15_TARGET','qpp-cpu'))
        data=dataset();scaler=fit_scaler(data['train']['raw']);w=initialize()
        checkpoint={'schema_version':1,'config':{'encoding':'angle','layers':1},'mapping':'positive_half',
                    'readout':'p1=(1-ZZ)/2','threshold':.5,'weights':w.tolist(),'scaler':scaler}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.json';path.write_text(json.dumps(checkpoint))
            restored=load_checkpoint(path)
            p,_=predict_proba(data['test']['raw'][:2],restored['weights'],restored['scaler'])
            ref,_=predict_proba(data['test']['raw'][:2],w,scaler,reference=True)
            np.testing.assert_allclose(p,ref,atol=1e-5)
            checkpoint['threshold']=.7;path.write_text(json.dumps(checkpoint))
            with self.assertRaises(ValueError):load_checkpoint(path)

if __name__=='__main__':unittest.main()
