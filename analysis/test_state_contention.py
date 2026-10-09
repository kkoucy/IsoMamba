import math
import unittest

import numpy as np
import torch

from analysis.analyze_state_contention import (
    MetricSamples,
    ScanTensors,
    analyze_isolated_recurrence,
    analyze_shared_recurrence,
    direction_sequences,
    merge_samples,
    reference_scan_full,
    stable_image_seed,
)


def _four_direction_scalar_tensors(values):
    values = torch.tensor(values, dtype=torch.float32)
    scan_x = values.view(1, 1, 1, -1).repeat(1, 4, 1, 1)
    length = scan_x.shape[-1]
    inverse_softplus_one = math.log(math.expm1(1.0))
    return ScanTensors(
        scan_x=scan_x,
        delta_raw=torch.zeros_like(scan_x),
        A=torch.zeros((4, 1, 1)),
        B=torch.ones((1, 4, 1, length)),
        C=torch.ones((1, 4, 1, length)),
        D=torch.zeros((4, 1)),
        delta_bias=torch.full((4, 1), inverse_softplus_one),
    )


class StateContentionTests(unittest.TestCase):
    def test_direction_sequences_matches_ss2d_order(self):
        spatial = torch.tensor([[[0, 1, 2], [3, 4, 5]]])
        expected = torch.tensor(
            [
                [
                    [0, 1, 2, 3, 4, 5],
                    [0, 3, 1, 4, 2, 5],
                    [5, 4, 3, 2, 1, 0],
                    [5, 2, 4, 1, 3, 0],
                ]
            ]
        )
        self.assertTrue(torch.equal(direction_sequences(spatial), expected))

    def test_stable_seed_is_repeatable_and_name_dependent(self):
        self.assertEqual(stable_image_seed(100, "a.png"), stable_image_seed(100, "a.png"))
        self.assertNotEqual(stable_image_seed(100, "a.png"), stable_image_seed(100, "b.png"))

    def test_reference_scan_uses_true_hidden_state(self):
        # softplus(inverse_softplus(1)) == 1; A=0 and B=1 gives a cumulative sum.
        scan_x = torch.tensor([[[[1.0, 2.0, 3.0]]]])
        inverse_softplus_one = math.log(math.expm1(1.0))
        tensors = ScanTensors(
            scan_x=scan_x,
            delta_raw=torch.zeros_like(scan_x),
            A=torch.zeros((1, 1, 1)),
            B=torch.ones((1, 1, 1, 3)),
            C=torch.ones((1, 1, 1, 3)),
            D=torch.zeros((1, 1)),
            delta_bias=torch.tensor([[inverse_softplus_one]]),
        )
        output, last_state = reference_scan_full(tensors)
        torch.testing.assert_close(output.flatten(), torch.tensor([1.0, 3.0, 6.0]))
        torch.testing.assert_close(last_state.flatten(), torch.tensor([6.0]))

    def test_shared_pair_split_and_isolated_boundaries(self):
        tensors = _four_direction_scalar_tensors([1.0, -1.0, 2.0, -2.0])
        labels = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, 4, 1)
        shared = analyze_shared_recurrence(tensors, eps=1e-8, labels=labels)
        for direction in range(4):
            self.assertEqual(shared[(direction, "intra")].eligible_pairs, 2)
            self.assertEqual(shared[(direction, "inter")].eligible_pairs, 1)

        isolated = analyze_isolated_recurrence(tensors, labels, num_groups=2, eps=1e-8)
        for direction in range(4):
            # Each two-token group contributes exactly one within-stream pair.
            self.assertEqual(isolated[(direction, "stream")].eligible_pairs, 2)

    def test_padding_breaks_statistical_adjacency_but_not_recurrence(self):
        tensors = _four_direction_scalar_tensors([1.0, 9.0, 2.0])
        valid = torch.tensor([[[True, False, True]]]).repeat(1, 4, 1)
        result = analyze_shared_recurrence(tensors, eps=1e-8, valid=valid)
        for direction in range(4):
            self.assertEqual(result[(direction, "stream")].eligible_pairs, 0)

    def test_near_zero_updates_are_excluded_from_cosine_only(self):
        tensors = _four_direction_scalar_tensors([0.0, 0.0, 1.0])
        result = analyze_shared_recurrence(tensors, eps=1e-8)
        for direction in range(4):
            samples = result[(direction, "stream")]
            self.assertEqual(samples.eligible_pairs, 2)
            self.assertEqual(samples.cosine_pairs, 0)
            self.assertEqual(samples.near_zero_pairs, 2)
            self.assertEqual(samples.volatility.size, 2)

    def test_merge_samples_preserves_counts_and_values(self):
        first = MetricSamples(np.array([0.5], np.float32), np.array([0.2], np.float32), 2, 1)
        second = MetricSamples(np.array([-0.5], np.float32), np.array([0.4], np.float32), 1, 0)
        merged = merge_samples([first, second])
        self.assertEqual(merged.eligible_pairs, 3)
        self.assertEqual(merged.near_zero_pairs, 1)
        np.testing.assert_allclose(merged.cosine, [0.5, -0.5])
        np.testing.assert_allclose(merged.volatility, [0.2, 0.4])


if __name__ == "__main__":
    unittest.main()
