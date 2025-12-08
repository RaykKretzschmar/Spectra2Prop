import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch
import os

def calculate_spearman_rank_correlation(predictions, targets):
    """
    Calculates the Spearman's Rank Correlation Coefficient between 
    predicted and actual values.
    
    Args:
        predictions (array-like): Predicted values (or logits/probabilities).
        targets (array-like): Ground truth values.
        
    Returns:
        float: The Spearman correlation coefficient (-1 to 1).
        float: The p-value associated with the correlation.
    """
    # Ensure inputs are 1D numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
        
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    
    # Calculate Spearman's rho
    rho, p_value = stats.spearmanr(predictions, targets)
    
    return rho, p_value

def plot_predicted_vs_actual(predictions, targets, save_path=None, title='Predicted vs Actual'):
    """
    Generates a scatter plot of Predicted vs Actual values.
    
    Args:
        predictions (array-like): Predicted values.
        targets (array-like): Ground truth values (Actual).
        save_path (str, optional): File path to save the plot (e.g., 'plots/scatter.png').
                                   If None, the plot is shown instead.
        title (str): Title of the plot.
    """
    # Ensure inputs are 1D numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
        
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    
    plt.figure(figsize=(8, 8))
    
    # Scatter plot
    plt.scatter(targets, predictions, alpha=0.5, c='blue', edgecolors='k', label='Data points')
    
    # Plot diagonal line (Perfect Prediction)
    min_val = min(np.min(targets), np.min(predictions))
    max_val = max(np.max(targets), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add correlation text to plot
    rho, _ = calculate_spearman_rank_correlation(predictions, targets)
    plt.text(0.05, 0.95, f"Spearman's ρ = {rho:.3f}", transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    if save_path:
        # Ensure directory exists
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # Test with dummy data if run directly
    print("Testing analysis module...")
    y_true = np.random.rand(100)
    y_pred = y_true + np.random.normal(0, 0.1, 100)
    
    rho, p = calculate_spearman_rank_correlation(y_pred, y_true)
    print(f"Spearman: {rho:.4f} (p={p:.4f})")
    
    plot_predicted_vs_actual(y_pred, y_true, title="Test Plot")