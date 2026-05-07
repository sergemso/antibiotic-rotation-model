"""
Tests for the regional antibiotic rotation and trojan pro-drug model.

Covers:
- Unit tests for all model functions (sync and async trojan)
- Regression tests for numerical stability
- Property-based tests for invariants
- Reference tests reproducing claims from the final manuscript

Author: Sergey Strebulaev
Date: 2026
License: CC BY 4.0
"""

import unittest
import numpy as np
from model import Parameters, ThreeRegionModel


class TestParameters(unittest.TestCase):
    """Tests for parameter dataclass defaults."""
    
    def test_default_values(self):
        """Verify default parameters match manuscript Table 3."""
        p = Parameters()
        self.assertEqual(p.s_growth, -0.005)
        self.assertEqual(p.c, 0.01)
        self.assertEqual(p.m_mig, 0.01)
        self.assertEqual(p.q_ext, 0.15)
        self.assertEqual(p.m_inter, 1e-4)
        self.assertEqual(p.gamma, 0.95)
        self.assertEqual(p.mu_esc, 1e-7)
        self.assertEqual(p.s_esc, -0.005)
        self.assertEqual(p.delta_T, 300)
        self.assertEqual(p.G, 100)


class TestRotationSchedule(unittest.TestCase):
    """Tests for the rotation schedule logic."""
    
    def setUp(self):
        self.model = ThreeRegionModel(Parameters())
    
    def test_one_region_active_at_all_times(self):
        """At any generation, exactly one region has active beta-lactams."""
        for t in range(900):
            active_count = sum(
                self.model._rotation_active(r, t) for r in range(3)
            )
            self.assertEqual(active_count, 1,
                             f"At t={t}, {active_count} regions active (expected 1)")
    
    def test_phase_duration(self):
        """Each region's active phase lasts exactly 300 generations (3 years)."""
        for region in range(3):
            active_gens = sum(
                self.model._rotation_active(region, t) for t in range(900)
            )
            self.assertEqual(active_gens, 300,
                             f"Region {region}: {active_gens} active gens (expected 300)")
    
    def test_cyclic_periodicity(self):
        """Rotation schedule repeats every 900 generations."""
        for t in range(900):
            for region in range(3):
                self.assertEqual(
                    self.model._rotation_active(region, t),
                    self.model._rotation_active(region, t + 900),
                    f"Periodicity broken at t={t}, region {region}"
                )
    
    def test_sync_trojan_impulse_timing(self):
        """Synchronous trojan impulses occur exactly every 300 generations."""
        for t in range(1, 1000):
            expected = (t % 300 == 0)
            actual = self.model._is_trojan_impulse_sync(t)
            self.assertEqual(expected, actual, f"Sync impulse mismatch at t={t}")
    
    def test_no_sync_impulse_at_t0(self):
        """Synchronous trojan is not applied at generation 0."""
        self.assertFalse(self.model._is_trojan_impulse_sync(0))
    
    def test_async_trojan_impulse_timing(self):
        """Asynchronous trojan: each region receives trojan at end of its active phase."""
        for region in range(3):
            for t in range(1, 901):
                expected = (t == (region + 1) * 300)
                actual = self.model._is_trojan_impulse_async(region, t)
                self.assertEqual(expected, actual,
                                 f"Async impulse mismatch: region={region}, t={t}")
    
    def test_no_async_impulse_at_t0(self):
        """Asynchronous trojan is not applied at generation 0."""
        for region in range(3):
            self.assertFalse(self.model._is_trojan_impulse_async(region, 0))


class TestModelDynamics(unittest.TestCase):
    """Tests for the core dynamical equations."""
    
    def setUp(self):
        self.params = Parameters()
        self.model = ThreeRegionModel(self.params)
    
    def test_q_stays_in_bounds(self):
        """Resistance frequency q must remain in [0, 1]."""
        for rotation in [True, False]:
            for trojan_sync in [True, False]:
                res = self.model.run_simulation(
                    n_generations=1000, rotation=rotation, trojan_sync=trojan_sync
                )
                q = res['q']
                self.assertTrue(np.all(q >= 0))
                self.assertTrue(np.all(q <= 1))
                self.assertFalse(np.any(np.isnan(q)))
    
    def test_r_stays_in_bounds(self):
        """Escape frequency r must remain in [0, 1]."""
        for rotation in [True, False]:
            for trojan_sync in [True, False]:
                res = self.model.run_simulation(
                    n_generations=1000, rotation=rotation, trojan_sync=trojan_sync
                )
                r = res['r']
                self.assertTrue(np.all(r >= 0))
                self.assertTrue(np.all(r <= 1))
                self.assertFalse(np.any(np.isnan(r)))
    
    def test_trojan_impulse_reduces_q(self):
        """After a trojan impulse, q must decrease (unless q=0)."""
        res = self.model.run_simulation(
            n_generations=600, rotation=True, trojan_sync=True
        )
        q = res['q']
        for region in range(3):
            q_before = q[region, 299]
            q_after = q[region, 300]
            if q_before > 1e-10:
                self.assertLess(q_after, q_before,
                    f"Region {region}: q increased after trojan impulse")
    
    def test_trojan_impulse_enriches_r_sync(self):
        """After a synchronous trojan impulse, escape frequency r must increase."""
        res = self.model.run_simulation(
            n_generations=600, rotation=True, trojan_sync=True
        )
        r = res['r']
        for region in range(3):
            r_before = r[region, 299]
            r_after = r[region, 300]
            if r_before > 1e-15:
                self.assertGreaterEqual(r_after, r_before,
                    f"Region {region}: r decreased after sync trojan impulse")
    
    def test_async_trojan_enriches_only_target_region(self):
        """Async trojan at t=300 should enrich Region A only."""
        params = Parameters()
        model = ThreeRegionModel(params)
        res = model.run_simulation(
            n_generations=350, rotation=True, trojan_sync=False
        )
        r = res['r']
        # At t=300, Region A receives trojan
        r_A_before = r[0, 299]
        r_A_after = r[0, 300]
        r_B_before = r[1, 299]
        r_B_after = r[1, 300]
        r_C_before = r[2, 299]
        r_C_after = r[2, 300]
        
        # Region A should be enriched
        if r_A_before > 1e-15:
            self.assertGreater(r_A_after, r_A_before,
                "Region A: escape frequency not enriched after async trojan")
        
        # Enrichment in A should be much larger than changes in B and C
        enrichment_A = r_A_after / r_A_before if r_A_before > 1e-15 else 1.0
        change_B = abs(r_B_after - r_B_before) / r_B_before if r_B_before > 1e-15 else 0.0
        change_C = abs(r_C_after - r_C_before) / r_C_before if r_C_before > 1e-15 else 0.0
        
        # Enrichment in A should be at least 10x larger than drift in B or C
        self.assertGreater(enrichment_A, 10 * max(change_B, change_C),
            f"Enrichment in A ({enrichment_A:.2e}) not >> changes in B ({change_B:.2e}), C ({change_C:.2e})")

class TestTimeToFailure(unittest.TestCase):
    """Tests for escape fixation detection."""
    
    def setUp(self):
        self.model = ThreeRegionModel(Parameters())
    
    def test_no_failure_when_r_low(self):
        """Time to failure returns n_generations when r never exceeds threshold."""
        r = np.full((3, 1000), 0.01)
        ttf = self.model.time_to_failure(r, threshold=0.50)
        self.assertEqual(ttf, 1000)
    
    def test_failure_when_two_regions_exceed(self):
        """Failure detected when at least 2 regions exceed threshold at same time."""
        r = np.full((3, 1000), 0.01)
        r[0, 150] = 0.60
        r[1, 150] = 0.55
        r[2, 150] = 0.01
        ttf = self.model.time_to_failure(r, threshold=0.50)
        self.assertEqual(ttf, 150)
    
    def test_no_failure_with_one_region(self):
        """Failure NOT detected when only 1 region exceeds threshold."""
        r = np.full((3, 1000), 0.01)
        r[0, 100] = 0.60
        ttf = self.model.time_to_failure(r, threshold=0.50)
        self.assertEqual(ttf, 1000)


class TestReproducibility(unittest.TestCase):
    """Regression tests for numerical stability."""
    
    def test_deterministic_output(self):
        """Same parameters produce identical results."""
        model1 = ThreeRegionModel(Parameters())
        model2 = ThreeRegionModel(Parameters())
        
        for rotation in [True, False]:
            for trojan_sync in [True, False]:
                res1 = model1.run_simulation(
                    n_generations=500, rotation=rotation, trojan_sync=trojan_sync
                )
                res2 = model2.run_simulation(
                    n_generations=500, rotation=rotation, trojan_sync=trojan_sync
                )
                np.testing.assert_array_almost_equal(res1['q'], res2['q'], decimal=12)
                np.testing.assert_array_almost_equal(res1['r'], res2['r'], decimal=12)


class TestManuscriptClaims(unittest.TestCase):
    """Reference tests reproducing specific claims from the final manuscript."""
    
    def test_claim_sync_no_protection(self):
        """
        Manuscript claim (Table 4): Under synchronous trojan,
        antibiotic rotation provides no protection (protection factor = 1.00).
        """
        params = Parameters()
        model = ThreeRegionModel(params)
        
        res_u = model.run_simulation(
            n_generations=50000, rotation=False, trojan_sync=True
        )
        res_r = model.run_simulation(
            n_generations=50000, rotation=True, trojan_sync=True
        )
        
        ttf_u = model.time_to_failure(res_u['r']) / params.G
        ttf_r = model.time_to_failure(res_r['r']) / params.G
        
        # Both should be ~21 years
        self.assertAlmostEqual(ttf_u, 21, delta=2)
        self.assertAlmostEqual(ttf_r, 21, delta=2)
        self.assertAlmostEqual(ttf_r / ttf_u, 1.0, delta=0.1)
    
    def test_claim_async_protection(self):
        """
        Manuscript claim (Table 6): Asynchronous trojan dramatically extends
        time to failure compared to synchronous trojan.
        """
        params = Parameters()
        model = ThreeRegionModel(params)
        
        # Sync (no rotation)
        res_sync = model.run_simulation(
            n_generations=50000, rotation=False, trojan_sync=True
        )
        ttf_sync = model.time_to_failure(res_sync['r']) / params.G
        
        # Async (no rotation)
        res_async = model.run_simulation(
            n_generations=50000, rotation=False, trojan_sync=False
        )
        ttf_async = model.time_to_failure(res_async['r']) / params.G
        
        # Async should be much larger than sync
        self.assertGreater(ttf_async, 10 * ttf_sync,
            f"Async TTF={ttf_async:.0f} not > 10x Sync TTF={ttf_sync:.0f}")
    
    def test_claim_rotation_unnecessary_for_async(self):
        """
        Manuscript claim: Once trojan is asynchronous, adding antibiotic
        rotation provides no additional benefit (protection factor = 1.00).
        """
        params = Parameters()
        model = ThreeRegionModel(params)
        
        res_u = model.run_simulation(
            n_generations=50000, rotation=False, trojan_sync=False
        )
        res_r = model.run_simulation(
            n_generations=50000, rotation=True, trojan_sync=False
        )
        
        ttf_u = model.time_to_failure(res_u['r']) / params.G
        ttf_r = model.time_to_failure(res_r['r']) / params.G
        
        # Both should be large (500+)
        self.assertGreater(ttf_u, 200)
        self.assertGreater(ttf_r, 200)
        # Protection factor should be ~1.0
        self.assertAlmostEqual(ttf_r / ttf_u, 1.0, delta=0.2,
            msg=f"Protection factor = {ttf_r/ttf_u:.2f}, expected ~1.0")
    
    def test_claim_s_esc_neutral_kills_sink_effect(self):
        """
        Manuscript claim (Table 7): When s_esc = 0, the sink effect vanishes.
        Async trojan TTF should be around 30 years.
        """
        params = Parameters(s_esc=0.0)
        model = ThreeRegionModel(params)
        
        res = model.run_simulation(
            n_generations=50000, rotation=False, trojan_sync=False
        )
        ttf = model.time_to_failure(res['r']) / params.G
        
        # Should be ~30 years, not 500+
        self.assertLess(ttf, 50,
            f"TTF={ttf:.0f} with s_esc=0, expected < 50 (sink effect absent)")
        self.assertGreater(ttf, 20,
            f"TTF={ttf:.0f} with s_esc=0, expected > 20")
    
    def test_claim_s_esc_negative_gives_full_protection(self):
        """
        Manuscript claim (Table 7): Any |s_esc| >= 0.005 gives full protection
        with async trojan (TTF 500+).
        """
        for s_esc in [-0.005, -0.010, -0.020]:
            with self.subTest(s_esc=s_esc):
                params = Parameters(s_esc=s_esc)
                model = ThreeRegionModel(params)
                
                res = model.run_simulation(
                    n_generations=50000, rotation=False, trojan_sync=False
                )
                ttf = model.time_to_failure(res['r']) / params.G
                
                self.assertEqual(ttf, 500,
                    f"s_esc={s_esc}: TTF={ttf:.0f}, expected 500 (no failure)")


class TestEdgeCases(unittest.TestCase):
    """Tests for extreme parameter values."""
    
    def test_gamma_zero_no_effect(self):
        """With gamma=0, trojan has no effect. q should not drop after impulse."""
        params = Parameters(gamma=0.0)
        model = ThreeRegionModel(params)
        
        res = model.run_simulation(
            n_generations=350, rotation=False, trojan_sync=True
        )
        q = res['q']
        # At t=300 (trojan impulse), q should not change much
        for region in range(3):
            self.assertAlmostEqual(q[region, 299], q[region, 300], delta=0.01)
    
    def test_high_migration_equalizes_regions(self):
        """Very high m_inter should equalize r across regions in async trojan."""
        params = Parameters(m_inter=1.0)
        model = ThreeRegionModel(params)
        
        res = model.run_simulation(
            n_generations=500, rotation=True, trojan_sync=False
        )
        r = res['r']
        
        # By t=500, all regions should have similar r
        std_dev = np.std(r[:, -1])
        self.assertLess(std_dev, 0.01,
            f"High migration: std of r across regions = {std_dev:.4f}")
    
    def test_no_escape_mutation(self):
        """With mu_esc=0, r should stay at initial value (zero)."""
        params = Parameters(mu_esc=0.0)
        model = ThreeRegionModel(params)
        
        # Force initial r to 0
        model.r_init = 0.0
        
        for trojan_sync in [True, False]:
            res = model.run_simulation(
                n_generations=1000, rotation=True, trojan_sync=trojan_sync
            )
            r = res['r']
            np.testing.assert_array_almost_equal(r, np.zeros_like(r), decimal=15)


def run_tests():
    """Run all tests with detailed output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    test_classes = [
        TestParameters,
        TestRotationSchedule,
        TestModelDynamics,
        TestTimeToFailure,
        TestReproducibility,
        TestManuscriptClaims,
        TestEdgeCases,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\nAll tests passed. Model implementation is consistent with manuscript claims.")
    else:
        print("\nSome tests failed. Check output above for details.")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)