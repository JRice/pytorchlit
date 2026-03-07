# Starting with a simple linear regression example using PyTorch to demonstrate the training process.
# `python minimal_train.py` => will show you the training process and the learned parameters after 100 epochs.
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Generate Synthetic Data: y = 2x + 1
data_points = 1000
X = torch.randn(data_points, 1).to(device)
Y = 2 * X + 1 + torch.randn(data_points, 1).to(device) * 0.1  # Added some noise

# 3. Define the Model
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1) # One input, one output

    def forward(self, x):
        return self.linear(x)

model = LinearModel().to(device)

# 4. Loss and Optimizer
criterion = nn.MSELoss() # Mean Squared Error
optimizer = optim.SGD(model.parameters(), lr=0.1) # Stochastic Gradient Descent
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5) # Learning rate decay

# 5. The Training Loop
for epoch in range(100):
    pred_y = model(X)
    loss = criterion(pred_y, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    scheduler.step()

    if (epoch + 1) % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, LR: {current_lr:.4f}')

# Check the learned parameters
[w, b] = model.parameters()
print(f"\nLearned Rule: y = {w.item():.2f}x + {b.item():.2f}")