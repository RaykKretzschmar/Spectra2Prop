import torch
from torch.utils.data import Dataset
from pyteomics import mgf
import numpy as np

class MGFDataset(Dataset):
    def __init__(self, mgf_file, max_peaks=100):
        self.mgf_file = mgf_file
        self.max_peaks = max_peaks
        # Index the file immediately. 
        # This builds a map of byte offsets without loading data to RAM.
        self.reader = mgf.IndexedMGF(mgf_file)

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        # Fetch specific spectrum from disk using byte offset
        spectrum = self.reader[idx]
        
        mz = spectrum['m/z array']
        intensity = spectrum['intensity array']
        
        # 1. Select top k peaks by intensity
        if len(mz) > self.max_peaks:
            # np.argpartition is faster than argsort for top-k selection
            top_indices = np.argpartition(intensity, -self.max_peaks)[-self.max_peaks:]
            mz = mz[top_indices]
            intensity = intensity[top_indices]
            
        # 2. Sort by m/z (Required because argpartition/argsort breaks order)
        sort_idx = np.argsort(mz)
        mz = mz[sort_idx]
        intensity = intensity[sort_idx]

        return {
            'mz': torch.tensor(mz, dtype=torch.float32),
            'intensity': torch.tensor(intensity, dtype=torch.float32)
        }

    def __getstate__(self):
        # Pickle protocol: Used if num_workers > 0
        # We generally shouldn't pickle the open file reader
        state = self.__dict__.copy()
        del state['reader']
        return state

    def __setstate__(self, state):
        # Restore state in worker process
        self.__dict__.update(state)
        self.reader = mgf.IndexedMGF(self.mgf_file)

def collate_fn(batch):    
    mz_list = [item['mz'] for item in batch]
    intensity_list = [item['intensity'] for item in batch]
    
    # Pad sequences
    mz_padded = torch.nn.utils.rnn.pad_sequence(mz_list, batch_first=True, padding_value=0.0)
    intensity_padded = torch.nn.utils.rnn.pad_sequence(intensity_list, batch_first=True, padding_value=0.0)
    
    # Create Attention Mask (True for real data, False for padding)
    # Shape: (batch_size, max_seq_len)
    lengths = torch.tensor([len(x) for x in mz_list])
    mask = torch.arange(mz_padded.size(1))[None, :] < lengths[:, None]
    
    return {
        'mz': mz_padded,
        'intensity': intensity_padded,
        'mask': mask
    }