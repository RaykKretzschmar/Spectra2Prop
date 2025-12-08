from torch.utils.data import DataLoader

import unittest
import torch
import os
import tempfile
import numpy as np
# Ensure this import path matches your project structure
from spectra2prop.data.dataloader import MGFDataset, collate_fn

class TestMGFDataset(unittest.TestCase):
    def setUp(self):
        # Create a temporary MGF file for testing
        self.temp_mgf = tempfile.NamedTemporaryFile(delete=False, suffix='.mgf', mode='w')
        
        # Spectrum 1: 2 peaks (Short sequence)
        # Sorted m/z: 100.0, 200.0
        self.temp_mgf.write("BEGIN IONS\nTITLE=Test1\nPEPMASS=500.0\nCHARGE=2+\n100.0 50.0\n200.0 100.0\nEND IONS\n")
        
        # Spectrum 2: 3 peaks (Longer sequence)
        # Sorted m/z: 150.0, 250.0, 350.0
        # Intensities: 20.0, 80.0, 40.0
        self.temp_mgf.write("BEGIN IONS\nTITLE=Test2\nPEPMASS=600.0\nCHARGE=3+\n150.0 20.0\n250.0 80.0\n350.0 40.0\nEND IONS\n")
        
        self.temp_mgf.close()

    def tearDown(self):
        # Clean up the temp file after tests run
        os.remove(self.temp_mgf.name)

    def test_collate_structure_and_mask(self):
        """
        Verifies that the batch has the correct shapes and that the 
        Attention Mask correctly identifies real data vs padding.
        """
        # Set max_peaks high enough so no cropping happens
        dataset = MGFDataset(self.temp_mgf.name, max_peaks=10)
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(loader))
        
        # 1. Check keys exist
        self.assertIn('mz', batch)
        self.assertIn('intensity', batch)
        self.assertIn('mask', batch, "Batch missing 'mask' key. Did you update collate_fn?")
        
        # 2. Check Shapes
        # Batch size 2, Max seq length 3 (from Spectrum 2)
        self.assertEqual(batch['mz'].shape, (2, 3))
        self.assertEqual(batch['mask'].shape, (2, 3))

        # 3. Verify Mask Logic
        # Spectrum 1 (index 0) has length 2. It should be [True, True, False]
        expected_mask_1 = torch.tensor([True, True, False])
        self.assertTrue(torch.equal(batch['mask'][0], expected_mask_1),
                        f"Mask failed for padded sequence. Got {batch['mask'][0]}")

        # Spectrum 2 (index 1) has length 3. It should be [True, True, True]
        expected_mask_2 = torch.tensor([True, True, True])
        self.assertTrue(torch.equal(batch['mask'][1], expected_mask_2))

    def test_data_integrity_and_padding(self):
        """
        Verifies that the actual m/z values are correct and that
        padding uses 0.0.
        """
        dataset = MGFDataset(self.temp_mgf.name, max_peaks=10)
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(loader))

        # Check Spectrum 1 (Short): Should be [100.0, 200.0, 0.0]
        # Using allclose for floating point comparisons
        expected_mz_1 = torch.tensor([100.0, 200.0, 0.0])
        self.assertTrue(torch.allclose(batch['mz'][0], expected_mz_1), 
                        f"Expected {expected_mz_1}, got {batch['mz'][0]}")

        # Check Spectrum 2 (Long): Should be [150.0, 250.0, 350.0]
        expected_mz_2 = torch.tensor([150.0, 250.0, 350.0])
        self.assertTrue(torch.allclose(batch['mz'][1], expected_mz_2))

    def test_top_k_filtering_logic(self):
        """
        Verifies that 'max_peaks' correctly selects the highest intensity peaks
        and then re-sorts them by m/z.
        """
        # Set max_peaks=1. Spectrum 2 has 3 peaks, so we expect cropping.
        dataset = MGFDataset(self.temp_mgf.name, max_peaks=1)
        
        # Get Spectrum 2 directly (skipping collate/padding for this specific check)
        item = dataset[1] 
        
        # Spectrum 2 Input Data:
        # m/z: [150.0, 250.0, 350.0]
        # Int: [ 20.0,  80.0,  40.0]
        
        # Logic Check:
        # 1. Highest intensity is 80.0 (corresponds to m/z 250.0)
        # 2. We keep only that one.
        
        self.assertEqual(len(item['mz']), 1, "Dataset failed to crop to max_peaks=1")
        
        # Check values
        self.assertAlmostEqual(item['mz'][0].item(), 250.0, places=4, 
                               msg="Failed to select the peak with highest intensity.")
        self.assertAlmostEqual(item['intensity'][0].item(), 80.0, places=4)

if __name__ == '__main__':
    unittest.main