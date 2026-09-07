import unittest
import cudaq
from noise import np,I,CHANNELS,OBS,kraus,density,exact,probabilities,sample

class NoiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cudaq.set_target('density-matrix-cpu')
    def test_cptp(self):
        for channel in CHANNELS:
            for p in (0,.3,.75,1):
                ks=kraus(channel,p);np.testing.assert_allclose(sum(k.conj().T@k for k in ks),I,atol=1e-12)
                rho=density(2,channel,p);self.assertAlmostEqual(np.trace(rho).real,1)
                self.assertGreaterEqual(np.linalg.eigvalsh(rho).min(),-1e-12)
    def test_analytic_channels(self):
        for p in (0,.3,.75,1):
            self.assertAlmostEqual(exact(0,'bit_flip',p,'Z0'),1-2*p)
            self.assertAlmostEqual(exact(1,'phase_flip',p,'X0'),1-2*p)
            self.assertAlmostEqual(exact(2,'depolarizing',p,'ZZ'),1-4*p/3)
    def test_phase_noise_not_visible_in_z(self):
        np.testing.assert_allclose(probabilities(2,'phase_flip',0,0),probabilities(2,'phase_flip',.5,0))
        self.assertAlmostEqual(exact(2,'phase_flip',.5,'XX'),0)
    def test_full_depolarization(self):
        np.testing.assert_allclose(density(2,'depolarizing',.75),np.eye(4)/4,atol=1e-12)
        self.assertAlmostEqual(exact(2,'depolarizing',1,'ZZ'),-1/3)
    def test_sampling_and_invalid(self):
        counts=sample(0,'bit_flip',1,0,128,42);self.assertEqual(counts,{'10':128})
        with self.assertRaises(ValueError):kraus('bit_flip',1.1)
        with self.assertRaises(ValueError):kraus('missing',.1)

if __name__=='__main__':unittest.main()
