"""Tests for Abhiram's Week 1 metric and adaptive helpers."""
import unittest
import random
from metrics import summarize_episodes
from adaptive import AdaptiveReplacement


class TestEvaluationHelpers(unittest.TestCase):
    def test_rates_and_averages(self):
        result = summarize_episodes([
            {'reward': 10, 'steps': 10, 'success': 1, 'collision': 0, 'unsafe_proposals': 2, 'interventions': 2},
            {'reward': -10, 'steps': 5, 'success': 0, 'collision': 1, 'unsafe_proposals': 3, 'interventions': 0},
        ])
        self.assertEqual(result['success_rate'], 0.5)
        self.assertEqual(result['collision_rate'], 0.5)
        self.assertEqual(result['average_reward'], 0)
        self.assertAlmostEqual(result['unsafe_proposal_rate'], 5 / 15)

    def test_adaptive_probability_increases_with_unsafe_rate(self):
        method = AdaptiveReplacement(window=4)
        for _ in range(4):
            method.record_proposal(False)
        low = method.random_probability()
        for _ in range(4):
            method.record_proposal(True)
        self.assertGreater(method.random_probability(), low)

    def test_adaptive_never_selects_unsafe_action(self):
        method = AdaptiveReplacement(window=4)
        rng = random.Random(42)
        for _ in range(50):
            self.assertIn(method.replace([1, 3], [100, 0, 100, 2], rng), [1, 3])


if __name__ == '__main__':
    unittest.main()
