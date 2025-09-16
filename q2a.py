import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math

# Fix randomness
torch.manual_seed(0)
np.random.seed(0)

# Define the target function
def target_fn(x):
    return 1.2 * np.sin(math.pi * x) - np.cos(2.4 * math.pi * x)

# Generate training and test datasets
x_train = np.arange(-2.0, 2.0 + 1e-9, 0.05)  # training set step 0.05
x_test = np.arange(-2.0, 2.0 + 1e-9, 0.01)   # test set step 0.01

y_train = np.array([target_fn(x) for x in x_train], dtype=np.float32)
y_test = np.array([target_fn(x) for x in x_test], dtype=np.float32)

x_train_t = torch.tensor(x_train.reshape(-1,1), dtype=torch.float32)
y_train_t = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32)
x_test_t = torch.tensor(x_test.reshape(-1,1), dtype=torch.float32)
y_test_t = torch.tensor(y_test.reshape(-1,1), dtype=torch.float32)

# Define a simple MLP: 1-n-1 architecture
class SimpleNet(nn.Module):
    def __init__(self, N=10):
        super(SimpleNet, self).__init__()
        self.hidden = nn.Linear(1, N)
        self.activation = nn.Tanh()
        self.output = nn.Linear(N, 1)
    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x

# Different hidden layer sizes to try
sizes = [1,2,3,4,5,6,7,8,9,10,20,50,100]

# Training parameters
learning_rate = 0.01
num_epochs = 300
results = []

# Prepare subplot grid (4 rows x 4 cols fits 13 plots nicely)
rows, cols = 4, 4
fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
axes = axes.flatten()

for idx, n in enumerate(sizes):
    model = SimpleNet(N=n)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training loop (sequential mode, batch size = 1)
    for epoch in range(num_epochs):
        for i in range(len(x_train_t)):
            xi = x_train_t[i].unsqueeze(0)
            yi = y_train_t[i].unsqueeze(0)
            y_pred = model(xi)
            loss = criterion(y_pred, yi)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluate after training
    with torch.no_grad():
        y_pred_test = model(x_test_t).numpy().reshape(-1)
        test_loss = np.mean((y_pred_test - y_test)**2)
        pred_m3 = model(torch.tensor([[-3.0]])).item()
        pred_p3 = model(torch.tensor([[3.0]])).item()
    
    results.append((n, test_loss, pred_m3, pred_p3))
    
    # Plot in the subplot
    ax = axes[idx]
    ax.plot(x_test, y_test, label="True", linewidth=1)
    ax.plot(x_test, y_pred_test, label="MLP", linewidth=1)
    ax.set_title(f"1-{n}-1")
    ax.legend(fontsize=6)

# Remove any unused subplots
for j in range(len(sizes), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Print summary
print("For x = -3: " + str(target_fn(-3)))
print("For x = 3: " + str(target_fn(3)))
for r in results:
    print(f"n =  {r[0]:3d} | f(-3)={r[2]:.4f}, f(3)={r[3]:.4f}")
