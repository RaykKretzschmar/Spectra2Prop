import unittest
import torch
import os
import tempfile
from torch.utils.data import DataLoader
from spectra2prop.data.dataloader import MGFDataset, collate_fn

class TestMGFDataset(unittest.TestCase):
    def setUp(self):
        self.temp_mgf = tempfile.NamedTemporaryFile(delete=False, suffix='.mgf', mode='w')
        self.temp_mgf.write("BEGIN IONS\nTITLE=Test1\nPEPMASS=500.0\nCHARGE=2+\n100.0 50.0\n200.0 100.0\nEND IONS\n")
        self.temp_mgf.write("BEGIN IONS\nTITLE=Test2\nPEPMASS=600.0\nCHARGE=3+\n150.0 20.0\n250.0 80.0\n350.0 40.0\nEND IONS\n")
        self.temp_mgf.close()

    def tearDown(self):
        os.remove(self.temp_mgf.name)

    def test_loading_and_collate(self):
        dataset = MGFDataset(self.temp_mgf.name, max_peaks=10)
        self.assertEqual(len(dataset), 2)
        
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(loader))
        
        self.assertIn('mz', batch)
        self.assertIn('intensity', batch)
        
        # Check shapes (batch_size, max_seq_len)
        # First spectrum len=2, second len=3 -> max_len should be 3
        self.assertEqual(batch['mz'].shape, (2, 3))
        self.assertEqual(batch['intensity'].shape, (2, 3))

if __name__ == '__main__':
    unittest.main()