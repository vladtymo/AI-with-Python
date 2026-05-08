import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target

print("Feature sample (raw):", X[0])

# 2. Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

print("Feature sample (normalized):", X[0])

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# 5. Define model
class IrisModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)  # logits
        )

    def forward(self, x):
        return self.net(x)

model = IrisModel()

# 6. Loss + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 7. Training loop
epochs = 20

for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()
    outputs = model(X_train)

    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 8. Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    predictions = torch.argmax(test_outputs, dim=1)
    accuracy = (predictions == y_test).float().mean()

print("Test accuracy:", accuracy.item())

# 9. Prediction on single sample
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
sample = scaler.transform(sample)
sample_tensor = torch.tensor(sample, dtype=torch.float32)

with torch.no_grad():
    logits = model(sample_tensor)
    pred_class = torch.argmax(logits, dim=1).item()

print("Logits:", logits.numpy())
print("Predicted class:", pred_class, "->", iris.target_names[pred_class])