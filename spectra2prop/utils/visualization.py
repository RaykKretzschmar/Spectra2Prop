import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.metrics import confusion_matrix, classification_report
import os

def visualize_embeddings(dataset, model, device, output_dir="notebooks"):
    """
    Generates UMAP projection of embeddings and saves to file.
    """
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_embeddings = []
    all_labels = []
    
    print("Extracting embeddings for visualization...")
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits, embeddings = model(inputs)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.array(all_labels)
    
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(all_embeddings)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embedding_2d[:, 0], 
        y=embedding_2d[:, 1], 
        hue=[dataset.idx_to_class[l] for l in all_labels],
        palette='viridis',
        s=50,
        alpha=0.7
    )
    plt.title("Spectra2Prop: UMAP of Internal Representations")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "umap_projection.png")
    plt.savefig(save_path)
    print(f"Saved UMAP plot to {save_path}")
