import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import main
from dataset.dataset_fs800 import FeatureDatasetWithStaticCache


class DatasetIntegrationTests(unittest.TestCase):
    def _build_dataset_root(self, root: Path, static_length: int = 2, static_dim: int = 768, static_dtype=np.float32):
        video_id = "sample"
        line = "sample 10 20 1 2 3 4 5 2\n"
        (root / "train_fs800.txt").write_text(line, encoding="utf-8")
        (root / "val_fs800.txt").write_text(line, encoding="utf-8")
        audio_dir = root / "new feature" / "ast_feature_fs1000_new"
        video_dir = root / "Timesformer_output_feature_fs800"
        cache_dir = root / "static_videomae_c1_cache"
        audio_dir.mkdir(parents=True)
        video_dir.mkdir(parents=True)
        cache_dir.mkdir(parents=True)
        np.save(audio_dir / (video_id + ".npy"), np.zeros((2, 768), dtype=np.float32))
        np.save(video_dir / (video_id + ".npy"), np.zeros((3, 15, 768), dtype=np.float32))
        np.save(
            cache_dir / "static_videomae_c1_sample_T2.npy",
            np.zeros((static_length, static_dim), dtype=static_dtype),
        )
        return cache_dir

    def test_loads_768_dimensional_cache_from_supplied_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_dir = self._build_dataset_root(root)

            dataset = FeatureDatasetWithStaticCache(
                root_path=str(root),
                is_train=True,
                cache_dir_name=str(cache_dir),
                cache_prefix="static_videomae_c1",
                static_feature_dim=768,
            )
            item = dataset[0]

            self.assertEqual(tuple(item[0].shape), (2, 768))
            self.assertEqual(tuple(item[1].shape), (3, 15, 768))
            self.assertEqual(tuple(item[9].shape), (2, 768))

    def test_rejects_static_cache_length_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_dir = self._build_dataset_root(root, static_length=3)
            dataset = FeatureDatasetWithStaticCache(
                root_path=str(root),
                is_train=True,
                cache_dir_name=str(cache_dir),
                cache_prefix="static_videomae_c1",
                static_feature_dim=768,
            )

            with self.assertRaisesRegex(ValueError, "static feature shape"):
                dataset[0]

    def test_converts_float16_retrieval_cache_to_float32(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_dir = self._build_dataset_root(
                root, static_dim=1536, static_dtype=np.float16
            )
            dataset = FeatureDatasetWithStaticCache(
                root_path=str(root),
                is_train=True,
                cache_dir_name=str(cache_dir),
                cache_prefix="static_videomae_c1",
                static_feature_dim=1536,
            )

            self.assertEqual(dataset[0][9].dtype, __import__("torch").float32)


class TrainingEntryPointTests(unittest.TestCase):
    def test_model_fuses_6914d_cross_attention_cache_before_static_projection(self):
        model = main.build_model()

        self.assertEqual(model.query_support_cross_attention.feature_dim, 768)
        self.assertEqual(model.static_proj.in_features, 768)
        self.assertEqual(model.static_proj.out_features, 128)

    def test_help_starts_without_referencing_undefined_device(self):
        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--static-cache-dir", result.stdout)
        self.assertIn("--static-feature-dim", result.stdout)


if __name__ == "__main__":
    unittest.main()
