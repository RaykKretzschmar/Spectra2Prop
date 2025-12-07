import unittest
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spectra2prop.models.transformer import SpectraTransformer

class TestSpectraTransformer(unittest.TestCase):
    def test_forward_pass(self):
        """Test that the model runs a forward pass and returns correct shape."""
        input_dim = 100
        d_model = 64
        nhead = 4
        num_layers = 2
        output_dim = 1
                                                            
        model = SpectraTransformer(input_dim, d_model, nhead, num_layers, output_dim)
                                                                            
        batch_size = 2
        # Create dummy input: (batch_size, sequence_length=1, input_dim)
        dummy_input = torch.randn(batch_size, 1, input_dim)
                                                                                                            
        output = model(dummy_input)
                                                                                                                            
        # Expected output shape: (batch_size, output_dim)
        self.assertEqual(output.shape, (batch_size, output_dim))

if __name__ == '__main__':
    unittest.main()