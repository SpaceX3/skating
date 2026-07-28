import argparse
import unittest

import torch

from main import (
    checkpoint_baseline_architecture,
    load_checkpoint,
    resolve_baseline_architecture,
)
from model import scoring_head


def tiny_corpus():
    return {
        "keys": torch.randn(3, 6),
        "video_ids": ["v0", "v1", "v2"],
        "instance_ids": ["i0", "i1", "i2"],
        "coarse_class_ids": torch.tensor([0, 1, 1]),
        "element_ids": torch.tensor([0, 1, 1]),
        "elements": ["<background>", "3A", "3A"],
        "goe_grades": torch.tensor([0.0, 1.0, -1.0]),
        "bvs": torch.tensor([0.0, 8.0, 8.0]),
        "panel_scores": torch.tensor([0.0, 9.0, 8.5]),
        "valid_score_mask": torch.tensor([False, True, True]),
        "coarse_class_vocab": ["background", "jump"],
        "element_vocab": ["<background>", "3A"],
        "metadata_version": "checkpoint-test-v1",
    }


def legacy_state_dict(static_proj_dim=128):
    return {
        "static_proj.weight": torch.zeros(static_proj_dim, 2048),
        "static_proj.bias": torch.zeros(static_proj_dim),
        "time_score_mlp.0.weight": torch.zeros(256, 512 + static_proj_dim),
        "time_score_mlp.0.bias": torch.zeros(256),
        "time_score_mlp.2.weight": torch.zeros(128, 256),
        "time_score_mlp.2.bias": torch.zeros(128),
        "time_score_mlp.4.weight": torch.zeros(1, 128),
        "time_score_mlp.4.bias": torch.zeros(1),
    }


class CheckpointArchitectureTest(unittest.TestCase):
    def test_raw_legacy_checkpoint_layout_is_inferred_from_tensors(self):
        state = {
            "module." + key: value for key, value in legacy_state_dict().items()
        }
        self.assertEqual(
            checkpoint_baseline_architecture(state),
            {
                "baseline_static_proj_dim": 128,
                "baseline_head_type": "legacy-womean",
            },
        )

    def test_metric_checkpoint_layout_is_inferred_from_tensors(self):
        state = {
            "static_proj.weight": torch.zeros(512, 2048),
            "time_score_mlp.0.weight": torch.ones(1024),
            "time_score_mlp.0.bias": torch.zeros(1024),
            "time_score_mlp.1.weight": torch.zeros(256, 1024),
        }
        self.assertEqual(
            checkpoint_baseline_architecture(state),
            {
                "baseline_static_proj_dim": 512,
                "baseline_head_type": "metric",
            },
        )

    def test_saved_checkpoint_layout_overrides_unspecified_cli_options(self):
        args = argparse.Namespace(
            baseline_static_proj_dim=None,
            baseline_head_type=None,
        )
        metadata = {
            "config": {
                "baseline_static_proj_dim": 128,
                "baseline_head_type": "legacy-womean",
            }
        }
        resolve_baseline_architecture(
            args,
            checkpoint_state=legacy_state_dict(),
            checkpoint_metadata=metadata,
            checkpoint_path="dynamic_best.pth",
        )
        self.assertEqual(args.baseline_static_proj_dim, 128)
        self.assertEqual(args.baseline_head_type, "legacy-womean")

    def test_incompatible_baseline_override_is_rejected(self):
        args = argparse.Namespace(
            baseline_static_proj_dim=512,
            baseline_head_type=None,
        )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            resolve_baseline_architecture(
                args,
                checkpoint_state=legacy_state_dict(),
                checkpoint_path="dynamic_best.pth",
            )

    def test_legacy_dynamic_state_loads_into_rag_model_without_ignored_baseline_keys(self):
        dynamic = scoring_head(
            depth=1,
            input_dim=4,
            dim=8,
            input_len=2,
            use_static_baseline=True,
            static_in_dim=6,
            baseline_static_proj_dim=4,
            baseline_head_type="legacy-womean",
        )
        rag = scoring_head(
            depth=1,
            input_dim=4,
            dim=8,
            input_len=2,
            use_static_branch=True,
            use_static_baseline=True,
            static_in_dim=6,
            static_proj_dim=8,
            baseline_static_proj_dim=4,
            baseline_head_type="legacy-womean",
            rag_corpus=tiny_corpus(),
        )
        metadata = load_checkpoint(
            rag,
            "legacy_dynamic_fixture.pth",
            device=torch.device("cpu"),
            strict=False,
            allowed_missing_prefixes=("rag.",),
            state_dict=dynamic.state_dict(),
            metadata={"training_stage": "dynamic"},
        )
        self.assertEqual(metadata["training_stage"], "dynamic")


if __name__ == "__main__":
    unittest.main()
