# Spectra2Prop

**Spectra2Prop** is a Deep Learning framework for the direct prediction of chemical properties (specifically chemical superclasses) from raw MS/MS spectra. 

Unlike traditional methods that rely on structural identification first, Spectra2Prop adopts a "structure-less" paradigm, utilizing a 1D Convolutional Neural Network (CNN) to learn features directly from binned spectral data.

## Features

- **End-to-End Learning**: Maps raw MS/MS spectra directly to chemical classes without intermediate structure prediction.
- **Spectral CNN**: A custom 1D-CNN designed to process binned spectral data efficiently.
- **Robust Data Handling**: Includes a custom PyTorch `Dataset` that handles MGF file loading, filtering (via `matchms`), and binning.
- **Visualization**: Tools to visualize learnable embeddings using UMAP, providing insights into how the model clusters different chemical classes.
- **Subset Compatibility**: Visualization tools support both full Datasets and PyTorch Subsets.

## Project Structure

```
Spectra2Prop/
├── data/               # Directory for storing datasets (e.g., MGF files)
├── notebooks/          # Jupyter notebooks for demonstration and experiments
│   └── showcase.ipynb  # Main walkthrough of the workflow
├── spectra2prop/       # Main package
│   ├── data/           # Data loading and processing modules
│   ├── models/         # Neural network models (SpectralCNN)
│   └── utils/          # Utility scripts (Visualization)
├── scripts/            # Helper scripts
└── README.md           # Project documentation
```

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/Spectra2Prop.git
    cd Spectra2Prop
    ```

2.  **Install Dependencies**:
    It is recommended to use a virtual environment (conda or venv).
    
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Key dependencies include `torch`, `matchms`, `numpy`, `matplotlib`, `seaborn`, `umap-learn`, and `tqdm`.*

## Usage

### Running the Showcase

The best way to get started is by running the **Showcase Notebook**:

```bash
jupyter notebook notebooks/showcase.ipynb
```

This notebook will guide you through:
1.  Downloading and loading the GNPS dataset.
2.  Visualizing raw input spectra.
3.  Initializing the `SpectralCNN` model.
4.  Training the model (or loading weights).
5.  Visualizing the learned embeddings using UMAP.

### Using the Package

You can also use the package components in your own scripts:

```python
import torch
from spectra2prop.data.dataset import SpectrumDataset
from spectra2prop.models.cnn1d import SpectralCNN

# 1. Load Data
# Ensure you have an MGF file at the specified path
dataset = SpectrumDataset("data/pesticides.mgf", mode='train')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# 2. Initialize Model
model = SpectralCNN(num_bins=2000, num_classes=len(dataset.class_to_idx))

# 3. Train (pseudo-code)
# ... standard PyTorch training loop ...
```

## Data

The project is designed to work with MGF (Mascot Generic Format) files. The default showcase uses the **GNPS NIH Natural Product Library**.

*Note: If automated download fails (e.g., due to server issues), please download the MGF file manually and place it in the `data/` directory.*

## License

[Apache License 2.0](LICENSE)
