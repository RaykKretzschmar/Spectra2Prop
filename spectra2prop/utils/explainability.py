import torch
import torch.nn.functional as F
import numpy as np

class GradCAM1D:
    """
    Implements 1D Grad-CAM (Gradient-weighted Class Activation Mapping) for
    spectral data analysis.

    This utility generates a heatmap indicating which regions of the input
    spectrum (wavenumbers) were most important for the model's prediction
    of a specific target class/property.
    """

    def __init__(self, model, target_layer=None):
        """
        Args:
            model (torch.nn.Module): The SpectralCNN model instance.
            target_layer (torch.nn.Module, optional): The specific 1D convolutional
                layer to hook. If None, it attempts to automatically find the
                last Conv1d layer in the model.
        """
        self.model = model.eval()
        self.target_layer = target_layer
        
        # Placeholders for hooks
        self.gradients = None
        self.activations = None

        # Automatically find the last Conv1d layer if not specified
        if self.target_layer is None:
            self.target_layer = self._find_last_conv_layer()

        # Register hooks
        self._register_hooks()

    def _find_last_conv_layer(self):
        """Iterates through model modules to find the last nn.Conv1d."""
        target = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv1d):
                target = module
        
        if target is None:
            raise ValueError("No Conv1d layer found in the model.")
        return target

    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to capture backward gradients."""
        # grad_output is a tuple, usually (tensor,)
        self.gradients = grad_output[0]

    def _save_activation(self, module, input, output):
        """Hook to capture forward activations."""
        self.activations = output

    def _register_hooks(self):
        """Registers the forward and backward hooks on the target layer."""
        self.target_layer.register_forward_hook(self._save_activation)
        # register_full_backward_hook is preferred in newer PyTorch versions
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def generate_heatmap(self, input_tensor, target_class_idx=None, normalize=True):
        """
        Generates the Grad-CAM heatmap for a given input spectrum.

        Args:
            input_tensor (torch.Tensor): Input spectrum with shape (1, 1, Length).
            target_class_idx (int, optional): The index of the target class to explain.
                If None, defaults to the class with the highest score.
            normalize (bool): If True, scales the output heatmap between 0 and 1.

        Returns:
            np.ndarray: The 1D heatmap resampled to the input length.
        """
        # Ensure gradients are zeroed
        self.model.zero_grad()
        
        # 1. Forward pass
        # The hooks will capture self.activations here
        output = self.model(input_tensor)

        # Determine target class
        if target_class_idx is None:
            target_class_idx = output.argmax(dim=1).item()

        # 2. Backward pass
        # Create a one-hot target tensor to backpropagate specific class score
        target_score = output[:, target_class_idx]
        target_score.backward(retain_graph=True)

        # 3. Compute Weights
        # self.gradients shape: (Batch, Channels, Length)
        # Global Average Pooling over the length dimension (dim=2)
        # Weights shape: (Batch, Channels, 1)
        weights = torch.mean(self.gradients, dim=2, keepdim=True)

        # 4. Generate weighted combination
        # self.activations shape: (Batch, Channels, Length)
        # We multiply weights * activations and sum across channels (dim=1)
        # Result shape: (Batch, 1, Length)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # 5. Apply ReLU
        # We only care about features that have a positive influence on the class of interest
        cam = F.relu(cam)

        # 6. Upsample to input size
        # Interpolate expects (Batch, Channel, Length)
        target_length = input_tensor.size(2)
        cam = F.interpolate(cam, size=target_length, mode='linear', align_corners=False)

        # Convert to numpy and flatten
        heatmap = cam.detach().cpu().numpy().flatten()

        # 7. Normalize (Optional but recommended for visualization)
        if normalize:
            if heatmap.max() > 0: # Avoid division by zero
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        return heatmap

    def remove_hooks(self):
        """Manually remove hooks if necessary to free memory."""
        # Note: In a production setting, you might store hook handles to call .remove()
        pass