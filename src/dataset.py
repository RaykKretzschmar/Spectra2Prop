import torch
from torch.utils.data import Dataset
import numpy as np

class SpectrumDataset(Dataset):
    def __init__(self, num_samples=100, num_bins=2000, max_mz=1000):
        """
        Args:
            num_samples (int): Number of synthetic samples to generate.
            num_bins (int): Resolution of the binned vector.
            max_mz (float): Maximum m/z value to consider.
        """
        self.num_samples = num_samples
        self.num_bins = num_bins
        self.max_mz = max_mz
        
        # Generate dummy data for demonstration
        # In production, load your list of spectra and labels here
        self.data = []
        for _ in range(num_samples):
            # Create random peaks: pairs of (mz, intensity)
            n_peaks = np.random.randint(10, 50)
            mz = np.random.uniform(50, max_mz, n_peaks)
            intensity = np.random.uniform(0, 1, n_peaks)
            
            # Dummy label (integer index for superclass)
            label = np.random.randint(0, 31) 
            self.data.append(((mz, intensity), label))

    def __len__(self):
        return self.num_samples

    def _bin_spectrum(self, mz, intensity):
        """
        Converts sparse peaks into a dense binned vector.
        """
        binned = np.zeros(self.num_bins, dtype=np.float32)
        bin_width = self.max_mz / self.num_bins
        
        # Quantize m/z values to bin indices
        bin_indices = np.floor(mz / bin_width).astype(int)
        
        # Clip indices that exceed the range
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
        
        # Sum intensities in each bin
        for idx, inten in zip(bin_indices, intensity):
            binned[idx] += inten
            
        # Optional: Log transform or Normalize (e.g., square root scaling)
        binned = np.sqrt(binned) 
        
        return binned

    def __getitem__(self, idx):
        (mz, intensity), label = self.data[idx]
        
        # 1. Process Spectrum into fixed vector
        vector = self._bin_spectrum(mz, intensity)
        
        # 2. Convert to Tensor
        # Shape must be (1, num_bins) for Conv1d input
        vector_tensor = torch.from_numpy(vector).unsqueeze(0) 
        
        return vector_tensor, torch.tensor(label, dtype=torch.long)
