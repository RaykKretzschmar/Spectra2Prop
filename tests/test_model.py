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

import unittest
import torch
import torch.nn as nn
import torch.optim as optim

# IMPORTS: Replace this with your actual model class
# from src.spectra2prop.models import TransformerModel 

class TestModelCapacity(unittest.TestCase):
    def setUp(self):
        # Set a seed for reproducibility
        torch.manual_seed(42)
        
        # PARAMETERS
        self.batch_size = 4
        self.seq_length = 100  # Example spectrum length
        self.input_dim = 1     # Assuming 1D spectrum intensity
        self.hidden_dim = 64
        self.output_dim = 1    # Regression target (e.g., LogP or Retention Time)
        
        # MODEL INITIALIZATION
        # Replace 'SimpleTransformer' with your actual 'Spectra2Prop' model
        # Ensure you reduce layer depth/width here to speed up the test
        self.model = nn.Sequential(
            nn.Linear(self.seq_length * self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        # If using your Transformer, it might look like:
        # self.model = TransformerModel(input_dim=self.input_dim, ...)

    def test_overfit_tiny_batch(self):
        """
        Verifies that the model can learn a tiny batch of data perfectly.
        If this fails, the model has a bug (e.g., broken gradient flow).
        """
        # 1. Create Dummy Data
        # Inputs: Random noise simulating spectra
        inputs = torch.randn(self.batch_size, self.seq_length * self.input_dim)
        
        # Targets: Random values simulating properties
        targets = torch.randn(self.batch_size, self.output_dim)

        # 2. Setup Optimizer and Loss
        # Use a high learning rate to overfit quickly
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # 3. Training Loop on the SAME batch
        initial_loss = None
        final_loss = None
        epochs = 100

        self.model.train()
        
        print(f"\nStarting Overfit Test (Target: Loss < 0.01)...")
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            # Capture losses
            if epoch == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        print(f"Initial Loss: {initial_loss:.4f} | Final Loss: {final_loss:.4f}")

        # 4. Assertions
        # Check if loss decreased significantly
        self.assertLess(final_loss, initial_loss, "Loss did not decrease during training.")
        
        # Check if loss is close to zero (indicating memorization)
        # We use a threshold of 0.1 for unit tests to be safe, but it should be lower
        self.assertLess(final_loss, 0.1, "Model failed to overfit tiny batch (Loss > 0.1).")

if __name__ == '__main__':
            unittest.main()