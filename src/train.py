import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.model import Spectra1DCNN
from src.dataset import SpectrumDataset

# --- Configuration ---
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 10
NUM_BINS = 2000
NUM_CLASSES = 31 # Approx number of ClassyFire superclasses
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Running on device: {DEVICE}")

    # 1. Prepare Data
    print("Loading data...")
    full_dataset = SpectrumDataset(num_samples=500, num_bins=NUM_BINS)
    
    # Split into train and val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Initialize Model
    model = Spectra1DCNN(input_dim=NUM_BINS, num_classes=NUM_CLASSES).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for spectra, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            spectra, labels = spectra.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(spectra)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        
        # Validation Step
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for spectra, labels in val_loader:
                spectra, labels = spectra.to(DEVICE), labels.to(DEVICE)
                outputs = model(spectra)
                
                # Get probabilities and predictions
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

    print("Training Complete.")
    
    # Save Model
    torch.save(model.state_dict(), "spectra2prop_model.pth")

if __name__ == "__main__":
    main()
