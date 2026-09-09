import unittest

import torch

from evaluate_robustness import apply_missingness
from utils.frame_drop import apply_training_frame_drop
from utils.losses import batched_pearson_loss, batched_ranknet_loss
from utils.mixed_modality_drop import apply_training_mixed_drop
from utils.whole_modality_drop import apply_training_whole_modality_drop

MODALITIES = ("visual", "text", "audio")


class TSMDTest(unittest.TestCase):
    def setUp(self):
        self.mask = torch.tensor([[True, True, True, True, True, False]])
        self.features = {
            modality: torch.ones(1, 6, 3) for modality in MODALITIES
        }
        self.video_ids = ["video"]

    def test_temporal_dropout_is_deterministic_and_preserves_padding(self):
        first, first_mask, _ = apply_training_frame_drop(
            self.features, self.mask, self.video_ids, "independent", [0.4], 42, 1
        )
        second, second_mask, _ = apply_training_frame_drop(
            self.features, self.mask, self.video_ids, "independent", [0.4], 42, 1
        )
        torch.testing.assert_close(first_mask, second_mask)
        self.assertFalse(first_mask[~self.mask].any())
        for modality in MODALITIES:
            torch.testing.assert_close(first[modality], second[modality])
            self.assertEqual(first[modality][0, 5].sum().item(), 3)

    def test_stream_dropout_removes_at_most_one_stream(self):
        _, drop_mask, states = apply_training_whole_modality_drop(
            self.features, self.mask, self.video_ids, "categorical", 0.0, 42, 1
        )
        self.assertLessEqual(len(states[0]), 1)
        self.assertLessEqual(drop_mask[0].any(dim=0).sum().item(), 1)

    def test_mixed_policy_selects_exactly_one_regime(self):
        _, drop_mask, states = apply_training_mixed_drop(
            self.features,
            self.mask,
            self.video_ids,
            (0.0, 1.0, 0.0),
            (0.4,),
            42,
            1,
            "independent",
        )
        self.assertEqual(states[0].regime, "temporal")
        self.assertFalse(drop_mask[~self.mask].any())

    def test_evaluation_synchronized_geometry(self):
        features = {name: value.clone() for name, value in self.features.items()}
        apply_missingness(features, self.mask, self.video_ids, "synchronized", 0.4, 42)
        for modality in MODALITIES[1:]:
            torch.testing.assert_close(features["visual"], features[modality])
        self.assertEqual((features["visual"][0, :5] == 0).all(dim=1).sum().item(), 2)
        self.assertEqual(features["visual"][0, 5].sum().item(), 3)

    def test_mpr_losses_are_finite(self):
        logits = torch.tensor([[2.0, 1.0, -1.0, -2.0]])
        scores = torch.sigmoid(logits)
        targets = torch.tensor([[1.0, 0.7, 0.3, 0.0]])
        mask = torch.ones_like(targets, dtype=torch.bool)
        self.assertTrue(torch.isfinite(batched_pearson_loss(scores, targets, mask)))
        self.assertTrue(torch.isfinite(batched_ranknet_loss(
            logits, targets, mask, num_pairs=16, min_gap=0.1, top_frac=0.5
        )))


if __name__ == "__main__":
    unittest.main()
