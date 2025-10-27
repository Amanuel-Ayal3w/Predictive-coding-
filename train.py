import jax.numpy as jnp
from jax import random
import argparse
import numpy as np
from sklearn.datasets import fetch_openml
from skimage.transform import resize 
import matplotlib.pyplot as plt
import os

# Assuming PCN_Simplified.py contains the PCN class
from PC_using_NGCLearn import PCN 

# --- 1. Data Loading and Preprocessing ---
def load_and_preprocess_mnist(n_samples, key):
    """
    Loads MNIST using fetch_openml, filters for classes '0' and '1', 
    downscales to 14x14, and returns JAX arrays.
    """
    # 1. Fetch the data (auto-downloads/caches using Scikit-learn)
    print("Fetching MNIST data (checks cache first)...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto', as_frame=False)
    
    # Preprocessing
    X_full = mnist.data.astype(np.float32) / 255.0 # Normalize features to [0, 1]
    Y_raw = mnist.target.astype(np.int32)
    
    # 2. Filter for classes 0 and 1
    mask = np.logical_or(Y_raw == 0, Y_raw == 1)
    X_filtered = X_full[mask]
    Y_filtered = Y_raw[mask]
    
    # 3. Downscale images (28x28 -> 14x14)
    target_dim = 14
    X_downscaled = np.array([
        resize(img.reshape(28, 28), (target_dim, target_dim), anti_aliasing=True).flatten()
        for img in X_filtered
    ])
    
    # 4. Convert labels to one-hot encoding (0 -> [1, 0], 1 -> [0, 1])
    N_filtered = len(Y_filtered)
    Y_onehot = np.zeros((N_filtered, 2), dtype=np.float32)
    Y_onehot[Y_filtered == 0, 0] = 1.0
    Y_onehot[Y_filtered == 1, 1] = 1.0

    # 5. Truncate and Shuffle
    N_use = min(n_samples, N_filtered)
    permutation = random.permutation(key, N_filtered)[:N_use]
    
    X_final = jnp.array(X_downscaled[permutation])
    Y_final = jnp.array(Y_onehot[permutation])
    
    return X_final, Y_final, X_filtered # Return X_filtered for raw visualization

# --- 2. Evaluation Function ---
def evaluate_model(model, X, Y):
    total_correct = 0
    total_loss = 0.0
    
    for x, y in zip(X, Y):
        x = x[None, :] 
        y = y[None, :]
        
        y_pred, _, efe = model.process(x, y, adapt_synapses=False)
        
        predicted_class = jnp.argmax(y_pred[0])
        true_class = jnp.argmax(y[0])
        
        if predicted_class == true_class:
            total_correct += 1
            
        total_loss += efe
            
    accuracy = total_correct / len(X)
    avg_loss = total_loss / len(X)
    return accuracy, avg_loss

# --- 3. Visualization Function (Saves to file) ---
def visualize_images(X_raw_784, Y_labels, n_images=5, filename="mnist_pc_visualization.jpeg"):
    """Takes raw 784-dim data, reshapes to 28x28, and saves the visualization."""
    
    # Take a random sample from the raw filtered data for visualization
    np.random.seed(42)
    indices = np.random.choice(len(X_raw_784), size=n_images, replace=False)
    
    X_sample = X_raw_784[indices]
    Y_sample = Y_labels[indices]
    
    X_reshaped = X_sample.reshape(-1, 28, 28)
    
    print(f"\nVisualizing {n_images} raw images from the dataset (28x28)...")
    
    fig, axes = plt.subplots(1, n_images, figsize=(n_images * 2.5, 3))
    
    if n_images == 1:
        axes = [axes]

    for i in range(n_images):
        ax = axes[i]
        ax.imshow(X_reshaped[i], cmap='gray_r', vmin=0, vmax=1) # Note: data is 0-1
        ax.set_title(f"Label: {Y_sample[i]}")
        ax.axis('off')

    plt.suptitle("Raw MNIST Digits (Classes 0 & 1)", fontsize=12)
    
    # Save the figure
    plt.savefig(filename, format='jpeg', dpi=200)
    plt.close(fig) 
    print(f"✅ Visualization saved successfully to: {filename}")

# --- 4. Plotting Function ---
def plot_training_history(history, model_name="PCN_Training_Curves"):
    epochs = range(1, len(history['train_efe']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epochs, history['train_efe'], label='Train EFE', color='tab:blue')
    ax1.plot(epochs, history['dev_efe'], label='Dev EFE', color='tab:orange')
    ax1.set_title('Expected Free Energy (EFE) over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('EFE (Loss)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.plot(epochs, history['dev_acc'], label='Dev Accuracy', color='tab:green', marker='.')
    ax2.set_title('Classification Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_yticks(np.arange(0.9, 1.01, 0.02))
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    filename = f"{model_name}_curves.jpeg"
    plt.tight_layout()
    plt.savefig(filename, format='jpeg', dpi=200)
    plt.close(fig)
    print(f"✅ Training curves saved to: {filename}")


# --- 5. Main Training Function ---
def train(args):
    key = random.PRNGKey(args.seed)
    
    # --- Data Loading ---
    # Load all data first to access the raw 28x28 data for visualization
    key_loader, key_train, key_dev = random.split(key, 3)
    
    # The loader key is the total number of filtered images available
    # We pass the full filtered size to get all relevant data for visualization
    _, _, X_raw_full = load_and_preprocess_mnist(100000, key_loader) 
    
    # Now load the actual subsets for training/dev using the correct counts
    train_X, train_Y, _ = load_and_preprocess_mnist(args.n_train, key_train)
    dev_X, dev_Y, _ = load_and_preprocess_mnist(args.n_dev, key_dev)
    
    # --- Visualization ---
    if args.visualize_count > 0:
        # Need to use the raw, un-normalized, un-downscaled data for plotting
        Y_raw_full = fetch_openml('mnist_784', version=1, parser='auto', as_frame=False).target.astype(np.int32)
        Y_raw_full_filtered = Y_raw_full[np.logical_or(Y_raw_full == 0, Y_raw_full == 1)]
        visualize_images(X_raw_full, Y_raw_full_filtered, args.visualize_count)

    # --- Model Configuration ---
    IN_DIM = train_X.shape[1] 
    OUT_DIM = train_Y.shape[1] 
    HIDDEN_DIMS = [50, 20]     
    key, model_key = random.split(key)
    
    model = PCN(dkey=model_key, in_dim=IN_DIM, out_dim=OUT_DIM, 
                hidden_dims=HIDDEN_DIMS, T=args.T_steps, eta=args.learning_rate,
                exp_dir="exp_mnist_direct")

    print(f"\n--- Starting PCN Training (MNIST 2-Class 14x14) ---")
    print(f"Model: {IN_DIM} -> {HIDDEN_DIMS} -> {OUT_DIM}")
    print(f"Train Samples: {len(train_X)}, Dev Samples: {len(dev_X)}, Epochs: {args.epochs}, T_steps: {args.T_steps}\n")

    history = {'train_efe': [], 'dev_efe': [], 'dev_acc': []}

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        key, perm_key = random.split(key)
        permutation = random.permutation(perm_key, len(train_X))
        shuffled_X = train_X[permutation]
        shuffled_Y = train_Y[permutation]
        
        train_loss = 0.0
        
        for i in range(len(shuffled_X)):
            x_sample = shuffled_X[i][None, :]
            y_sample = shuffled_Y[i][None, :]
            _, _, efe = model.process(x_sample, y_sample, adapt_synapses=True)
            train_loss += efe
        
        avg_train_loss = train_loss / len(shuffled_X)
        
        dev_acc, dev_loss = evaluate_model(model, dev_X, dev_Y)
        
        history['train_efe'].append(float(avg_train_loss))
        history['dev_efe'].append(float(dev_loss))
        history['dev_acc'].append(float(dev_acc))
        
        print(f"Epoch {epoch:03d}/{args.epochs} | Train EFE: {avg_train_loss:.4f} | Dev EFE: {dev_loss:.4f} | Dev Acc: {dev_acc:.4f}")

    # Final Steps
    plot_training_history(history)
    print("\n--- Training Complete ---")
    print(f"Final Dev Accuracy: {history['dev_acc'][-1]:.4f}")


# --- 6. Command Line Interface (Simplified) ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a simplified PCN on a subset of MNIST.")
    
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--T_steps', type=int, default=10, help="Time steps for PCN inference (E-step).")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate (eta).")
    parser.add_argument('--n_train', type=int, default=1000, help="Number of training samples.")
    parser.add_argument('--n_dev', type=int, default=200, help="Number of development/validation samples.")
    parser.add_argument('--visualize_count', type=int, default=5, help="Number of images to visualize (set to 0 to skip).")
    
    args = parser.parse_args()
    train(args)