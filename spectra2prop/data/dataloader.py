import torch
from torch.utils.data import Dataset
from pyteomics import mgf
import numpy as np

class MGFDataset(Dataset):
    def __init__(self, mgf_file, max_peaks=100):
        self.mgf_data = list(mgf.read(mgf_file))
        self.max_peaks = max_peaks

    def __len__(self):
        return len(self.mgf_data)

    def __getitem__(self, idx):
        spectrum = self.mgf_data[idx]
        mz = spectrum['m/z array']
        intensity = spectrum['intensity array']
        
        # Select top k peaks
        if len(mz) > self.max_peaks:
            top_indices = np.argsort(intensity)[-self.max_peaks:]
            mz = mz[top_indices]
            intensity = intensity[top_indices]
            
        # Sort by m/z for consistency
        sort_idx = np.argsort(mz)
        mz = mz[sort_idx]
        intensity = intensity[sort_idx]

        return {
            'mz': torch.tensor(mz, dtype=torch.float32),
            'intensity': torch.tensor(intensity, dtype=torch.float32)
        }

    mz_list = [item['mz'] for item in batch]
    intensity_list = [item['intensity'] for item in batch]
    
    # Pad sequences
    mz_padded = torch.nn.utils.rnn.pad_sequence(mz_list, batch_first=True, padding_value=0.0)
    intensity_padded = torch.nn.utils.rnn.pad_sequence(intensity_list, batch_first=True, padding_value=0.0)
    
    return {
        'mz': mz_padded,
        'intensity': intensity_padded
    }