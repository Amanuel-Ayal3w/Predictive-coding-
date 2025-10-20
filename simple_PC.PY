import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)

# Hyperparameters
input_size = 3
hidden_size = 4
output_size = 2
eta = 0.05
inference_steps = 50

# Hidden -> Input
W1 = torch.randn(hidden_size, input_size) * 0.1
b1 = torch.zeros(input_size)

# Output -> Hidden
W2 = torch.randn(hidden_size, output_size) * 0.1
b2 = torch.zeros(hidden_size)

# Example input batch (unsupervised, batch size 2)
x_input = torch.tensor([[0.2, 0.5, 0.1],
                        [0.9, 0.3, 0.7]], dtype=torch.float)

# Initialize states randomly
h = torch.randn(x_input.shape[0], hidden_size)
y = torch.randn(x_input.shape[0], output_size)

# Activation function
def relu(x):
    return x.clamp(min=0)

def relu_deriv(x):
    return (x > 0).float()

# Lists to store errors
hidden_errors = []
input_errors = []

# Inference loop
for step in range(inference_steps):
    h_pred = y @ W2.T + b2
    x_pred = h @ W1 + b1

    e_hidden = h - h_pred
    e_input = x_input - x_pred

    hidden_errors.append(e_hidden.abs().mean().item())
    input_errors.append(e_input.abs().mean().item())

    y -= eta * (e_hidden @ W2) * relu_deriv(y)
    h -= eta * (e_hidden + e_input @ W1.T) * relu_deriv(h)

# Weight updates (local)
W2 += eta * e_hidden.T @ y
b2 += eta * e_hidden.sum(0)
W1 += eta * h.T @ e_input
b1 += eta * e_input.sum(0)

# Print final weights
print("Updated W1:\n", W1)
print("Updated W2:\n", W2)

# Plot errors
plt.plot(range(1, inference_steps+1), hidden_errors, marker='o', label='Hidden Error')
plt.plot(range(1, inference_steps+1), input_errors, marker='s', label='Input Error')
plt.xlabel('Inference Step')
plt.ylabel('Mean Absolute Error')
plt.title('Predictive Coding Error Relaxation')
plt.legend()

# Save plot as JPEG
plt.savefig("predictive_coding_errors.jpeg", format='jpeg')
print("Plot saved as predictive_coding_errors.jpeg")
