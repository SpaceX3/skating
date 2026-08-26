import tempfile
import unittest
from pathlib import Path

import torch

import main


class DynamicWarmStartTests(unittest.TestCase):
    def test_loads_only_dynamic_parameters(self):
        model = main.build_model()
        before = {
            key: value.clone()
            for key, value in model.state_dict().items()
            if key.startswith(("static_proj.", "time_score_mlp."))
        }
        source = {}
        for key, value in model.state_dict().items():
            if key == "static_proj.weight":
                source[key] = torch.full((128, 2048), 9.0)
            elif key.startswith(("static_proj.", "time_score_mlp.")):
                source[key] = torch.full_like(value, 9.0)
            else:
                source[key] = torch.full_like(value, 3.0)

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pth"
            torch.save(source, checkpoint)

            loaded_keys = main.load_dynamic_checkpoint(model, str(checkpoint))

        self.assertEqual(len(loaded_keys), 34)
        for key, value in model.state_dict().items():
            if key.startswith(("static_proj.", "time_score_mlp.")):
                self.assertTrue(torch.equal(value, before[key]), key)
            else:
                self.assertTrue(torch.equal(value, torch.full_like(value, 3.0)), key)


if __name__ == "__main__":
    unittest.main()
