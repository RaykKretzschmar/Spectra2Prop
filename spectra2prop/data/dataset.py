import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from spectra2prop.data.loader import load_and_process_spectra, download_data
import collections

class SpectrumDataset(Dataset):
    def __init__(self, file_path, num_bins=2000, max_mz=1000, mode='train'):
        """
        Args:
            file_path (str): Path to MGF file (optional if mode='synthetic').
            num_bins (int): Resolution of the binned vector.
            max_mz (float): Maximum m/z.
            mode (str): 'train', 'test', or 'synthetic'.
        """
        self.num_bins = num_bins
        self.max_mz = max_mz
        
        if mode == 'synthetic':
            self._generate_synthetic_data(num_samples=1000)
            return

        if file_path.startswith("http"):
             filename = file_path.split("/")[-1]
             if not filename.endswith(".mgf"):
                 filename = "downloaded_data.mgf"
             
             local_path = os.path.join("data", filename)
             os.makedirs("data", exist_ok=True)
             
             print(f"URL detected. Downloading to {local_path}...")
             download_data(file_path, local_path)
             file_path = local_path
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        all_spectra = load_and_process_spectra(file_path)
        
        self.spectra = []
        self.labels = []
        self.classes = set()
        
        print("Extracting labels...")
        valid_spectra = []
        for s in all_spectra:
            meta = s.metadata
            label = meta.get('superclass') or meta.get('compound_class') or meta.get('class') or meta.get('compound_name')
            
            if label:
                valid_spectra.append(s)
                self.classes.add(label)
                self.labels.append(label)
            else:
                 pass
        
        self.class_to_idx = {c: i for i, c in enumerate(sorted(list(self.classes)))}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        
        print(f"Found {len(self.classes)} unique classes: {list(self.classes)[:5]}...")
        
        counts = collections.Counter(self.labels)
        common_classes = {c for c, count in counts.items() if count >= 1}
        
        self.data = []
        self.targets = []
        
        for s, l in zip(valid_spectra, self.labels):
            if l in common_classes:
                self.data.append(s)
                self.targets.append(self.class_to_idx[l])
        
        n = len(self.data)
        indices = np.random.RandomState(42).permutation(n)
        split = int(0.8 * n)
        if mode == 'train':
            self.indices = indices[:split]
        else:
            self.indices = indices[split:]
            
        print(f"Dataset ({mode}): {len(self.indices)} samples.")

    def _generate_synthetic_data(self, num_samples):
        print("Generating synthetic data...")
        self.data = []
        self.targets = []
        self.class_to_idx = {
            "Lipids": 0, "Organic Acids": 1, "Benzenoids": 2, 
            "Organoheterocyclics": 3, "Phenylpropanoids": 4
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = set(self.class_to_idx.keys())
        
        for _ in range(num_samples):
            label_idx = np.random.randint(0, len(self.class_to_idx))
            
            n_peaks = np.random.randint(10, 50)
            
            if label_idx == 0:
                mz = np.random.normal(200, 50, n_peaks)
            elif label_idx == 1:
                mz = np.random.normal(800, 50, n_peaks)
            elif label_idx == 2:
                mz = np.random.uniform(100, 900, n_peaks)
            elif label_idx == 3:
                mz = np.concatenate([np.random.normal(300, 20, n_peaks//2), np.random.normal(600, 20, n_peaks - n_peaks//2)])
            else:
                 mz = np.random.uniform(50, 400, n_peaks)

            mz = np.clip(mz, 0, self.max_mz - 1)
            intensity = np.random.uniform(0.1, 1.0, n_peaks)
            
            MockSpectrum = collections.namedtuple('MockSpectrum', ['peaks'])
            MockPeaks = collections.namedtuple('MockPeaks', ['mz', 'intensities'])
            
            self.data.append(MockSpectrum(peaks=MockPeaks(mz, intensity)))
            self.targets.append(label_idx)
            
        self.indices = np.arange(num_samples)
        print(f"Generated {num_samples} synthetic spectra.")


    def __len__(self):
        return len(self.indices)

    def _bin_spectrum(self, mz, intensity):
        binned = np.zeros(self.num_bins, dtype=np.float32)
        bin_width = self.max_mz / self.num_bins
        bin_indices = np.floor(mz / bin_width).astype(int)
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
        for idx, inten in zip(bin_indices, intensity):
            binned[idx] += inten
        return np.sqrt(binned)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        spectrum = self.data[real_idx]
        target = self.targets[real_idx]
        
        mz, intensity = spectrum.peaks.mz, spectrum.peaks.intensities
        vector = self._bin_spectrum(mz, intensity)
        
        return torch.from_numpy(vector).unsqueeze(0), torch.tensor(target, dtype=torch.long)
