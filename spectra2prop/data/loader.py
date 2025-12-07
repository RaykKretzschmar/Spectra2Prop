import os
import requests
from matchms.importing import load_from_mgf
from matchms.filtering import default_filters, normalize_intensities, select_by_mz, select_by_relative_intensity
from tqdm import tqdm

def download_data(url, save_path):
    """
    Downloads data from a generic URL.
    """
    if os.path.exists(save_path):
        print(f"File already exists at {save_path}")
        return

    print(f"Downloading data from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=1024), unit="KB"):
                f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        raise e

def clean_metadata(spectrum):
    return spectrum

def load_and_process_spectra(file_path):
    """
    Loads spectra from MGF and applies filters.
    """
    print("Loading spectra...")
    spectra = list(load_from_mgf(file_path))
    
    print(f"Loaded {len(spectra)} spectra. Applying filters...")
    
    processed_spectra = []
    for spectrum in tqdm(spectra):
        spectrum = default_filters(spectrum)
        
        spectrum = normalize_intensities(spectrum)
        
        spectrum = select_by_relative_intensity(spectrum, intensity_from=0.01)
        spectrum = select_by_mz(spectrum, mz_from=10, mz_to=1000)
        
        if spectrum:
            processed_spectra.append(spectrum)
            
    print(f"Retained {len(processed_spectra)} spectra after filtering.")
    return processed_spectra

if __name__ == "__main__":
    url = "https://gnps-external.ucsd.edu/gnpslibrary/GNPS-NIH-NATURAL-PRODUCT-LIBRARY.mgf"
    save_path = "data/GNPS-NIH-NATURAL-PRODUCT-LIBRARY.mgf"
    os.makedirs("data", exist_ok=True)
    
    download_data(url, save_path)
    spectra = load_and_process_spectra(save_path)
    
    if spectra:
        print("Example Metadata:", spectra[0].metadata)
