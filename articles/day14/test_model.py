import os
import unittest
from model import cudaq,np,Config,initialize,predict,predict_batch,reference_prediction,arguments

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cudaq.set_target(os.environ.get('DAY14_TARGET','qpp-cpu'))

    def test_config_and_input_contracts(self):
        for args in [('bad',1),('angle',0),('amplitude',4)]:
            with self.assertRaises(ValueError):Config(*args)
        for features,weights,config in [([3,4],initialize(),Config('amplitude')),([0,0,0],initialize(),Config('amplitude')),([2,0],initialize(),Config()),([0,0],[0]*3,Config())]:
            with self.assertRaises(ValueError):arguments(features,weights,config)
        with self.assertRaises(ValueError):predict_batch([],initialize())

    def test_composition_against_numpy(self):
        for encoding,x in [('angle',[.25,-.4]),('amplitude',[1,-2,3,-4])]:
            for layers in (1,2,3):
                config=Config(encoding,layers);w=initialize(layers)
                self.assertAlmostEqual(predict(x,w,config),reference_prediction(x,w,config),delta=1e-5)

    def test_batch_and_input_immutability(self):
        x=np.array([[.2,-.4],[.6,.3]]);w=initialize();before=x.copy();wb=w.copy()
        batch=predict_batch(x,w)
        np.testing.assert_allclose(batch,[predict(row,w) for row in x],atol=1e-5)
        np.testing.assert_array_equal(x,before);np.testing.assert_array_equal(w,wb)

    def test_zero_weights_retain_cnot(self):
        # Positive-half angle x=[0,1] prepares |+1>; one CNOT changes ZZ from 0 to -1.
        self.assertAlmostEqual(predict([0,1],[0]*4),-1.,delta=1e-5)
        self.assertAlmostEqual(predict([0,1],[0]*8,Config('angle',2)),0.,delta=1e-5)

    def test_parameter_update_changes_both_models(self):
        for encoding,x in [('angle',[.25,-.4]),('amplitude',[1,-2,3,-4])]:
            config=Config(encoding);w=initialize();changed=w.copy();changed[1]+=.5
            self.assertGreater(abs(predict(x,w,config)-predict(x,changed,config)),1e-3)

    def test_common_ansatz_preserves_encoding_collision(self):
        w=initialize();config=Config('amplitude')
        self.assertAlmostEqual(predict([3,4,0],w,config),predict([30,40,0],w,config),delta=1e-5)

if __name__=='__main__':unittest.main()
