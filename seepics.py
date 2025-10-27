import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from jax import random, numpy as jnp

# --- 1. Data Loading and Filtering ---
def load_and_visualize_mnist(n_images=8):
    """
    Loads MNIST from cache, filters for classes '0' and '1', and prepares 
    raw 28x28 images for visualization.
    """
    # 1. Fetch data. Sklearn automatically checks your cache (~/scikit_learn_data)
    mnist = fetch_openml('mnist_784', version=1, parser='auto', as_frame=False)
    
    X_full = mnist.data.astype(np.float32) 
    Y_full = mnist.target.astype(np.int32)
    
    # 2. Filter for classes 0 and 1 (to match training data)
    mask = np.logical_or(Y_full == 0, Y_full == 1)
    X_filtered = X_full[mask]
    Y_filtered = Y_full[mask]
    
    # 3. Take a small sample (using a fixed seed for consistency)
    np.random.seed(42)
    indices = np.random.choice(len(X_filtered), size=n_images, replace=False)
    
    X_sample = X_filtered[indices]
    Y_sample = Y_filtered[indices]

    # 4. Reshape from 784-vector to 28x28 matrix for plotting
    X_reshaped = X_sample.reshape(-1, 28, 28)
    
    return X_reshaped, Y_sample

# --- 2. Visualization Function (Modified to Save File) ---
def visualize_images(X_matrices, Y_labels, filename="mnist_sample.jpeg"):
    """Creates the plot and saves the figure as a JPEG file."""
    num_images = len(X_matrices)
    
    # Define the figure and subplots
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2.5, 3))
    
    if num_images == 1:
        axes = [axes]

    for i in range(num_images):
        ax = axes[i]
        # Display the image matrix
        ax.imshow(X_matrices[i], cmap='gray_r', vmin=0, vmax=255)
        ax.set_title(f"Label: {Y_labels[i]}")
        ax.axis('off')

    plt.suptitle("Raw MNIST Digits (Classes 0 & 1) from Cache", fontsize=12)
    
    # NEW: Save the figure instead of showing it interactively
    plt.savefig(filename, format='jpeg', dpi=150)
    plt.close(fig) # Close the figure to free up memory
    
    print(f"\nVisualization saved successfully to: {filename}")
    print("You can view the file in your current directory.")


# --- 3. Main Execution ---
if __name__ == '__main__':
    NUM_SAMPLES = 8 
    
    images, labels = load_and_visualize_mnist(NUM_SAMPLES)
    visualize_images(images, labels, filename="mnist_pc_visualization.jpeg")