import unittest

import torch

import main
from model import QueryMemoryGatedFusion


class QueryMemoryGatedFusionTests(unittest.TestCase):
    def test_keeps_query_as_residual_and_gates_memory_update(self):
        fusion = QueryMemoryGatedFusion(feature_dim=2, gate_hidden_dim=4)
        with torch.no_grad():
            fusion.memory_proj.weight.copy_(torch.eye(2))
            fusion.memory_proj.bias.zero_()
            for parameter in fusion.gate_mlp.parameters():
                parameter.zero_()

        query = torch.tensor([[[1.0, 0.0]]])
        memory = torch.tensor([[[0.0, 1.0]]])
        output = fusion(torch.cat((query, memory), dim=-1))
        expected = fusion.output_norm(query + 0.5 * memory)

        torch.testing.assert_close(output, expected)

    def test_training_model_projects_fused_768d_representation(self):
        model = main.build_model(static_in_dim=1536)

        self.assertEqual(model.query_memory_fusion.feature_dim, 768)
        self.assertEqual(model.static_proj.in_features, 768)


if __name__ == "__main__":
    unittest.main()
