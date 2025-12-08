import unittest
import torch
import torch.nn as nn
import numpy as np
from spectra2prop.utils.explainability import GradCAM1D

# --- Dummy Model for Testing ---
class DummySpectralCNN(nn.Module):
    """
    A simple 1D CNN to simulate the structure of the real SpectralCNN
    for unit testing purposes.
    """
    def __init__(self):
        super().__init__()
        # First conv layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # Second conv layer (This should be the auto-detected target)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class NoConvModel(nn.Module):
    """A model with no Conv1d layers to test error handling."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.linear(x)

# --- Test Case ---
class TestGradCAM1D(unittest.TestCase):
    
    def setUp(self):
        # Initialize model and dummy input
        self.model = DummySpectralCNN()
        self.input_len = 100
        # Shape: (Batch=1, Channels=1, Length=100)
        self.input_tensor = torch.randn(1, 1, self.input_len, requires_grad=True)

    def test_initialization_auto_layer(self):
        """Test if GradCAM automatically finds the last Conv1d layer."""
        grad_cam = GradCAM1D(self.model)
        
        # Should be an instance of Conv1d
        self.assertIsInstance(grad_cam.target_layer, nn.Conv1d)
        # Should specifically be conv2 (the last one defined/used)
        self.assertEqual(grad_cam.target_layer, self.model.conv2)

    def test_initialization_manual_layer(self):
        """Test if GradCAM accepts a user-specified layer."""
        grad_cam = GradCAM1D(self.model, target_layer=self.model.conv1)
        
        self.assertEqual(grad_cam.target_layer, self.model.conv1)

    def test_initialization_no_conv_error(self):
        """Test if ValueError is raised when no Conv1d layer exists."""
        model_no_conv = NoConvModel()
        with self.assertRaises(ValueError):
            GradCAM1D(model_no_conv)

    def test_heatmap_shape(self):
        """Test if the generated heatmap matches the input length."""
        grad_cam = GradCAM1D(self.model)
        heatmap = grad_cam.generate_heatmap(self.input_tensor, target_class_idx=0)
        
        # Check type
        self.assertIsInstance(heatmap, np.ndarray)
        # Check shape (should be flattened 1D array of input length)
        self.assertEqual(heatmap.shape, (self.input_len,))

    def test_heatmap_values_normalized(self):
        """Test if heatmap values are normalized between 0 and 1."""
        grad_cam = GradCAM1D(self.model)
        heatmap = grad_cam.generate_heatmap(self.input_tensor, normalize=True)
        
        self.assertGreaterEqual(heatmap.min(), 0.0)
        # Allow small floating point tolerance
        self.assertLessEqual(heatmap.max(), 1.0 + 1e-6)

    def test_hooks_registration(self):
        """Test that hooks are registered correctly."""        
        grad_cam = GradCAM1D(self.model)
        self.assertTrue(len(grad_cam.handles) > 0, "Hooks handles not stored")


    def test_remove_hooks_api(self):
        """Test if the remove_hooks method runs without error."""
        grad_cam = GradCAM1D(self.model)
        try:
            grad_cam.remove_hooks()
        except Exception as e:
            self.fail(f"remove_hooks() raised {type(e).__name__} unexpectedly!")

if __name__ == '__main__':
    unittest.main()