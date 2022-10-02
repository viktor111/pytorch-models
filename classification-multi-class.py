from pathlib import Path
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs 
from sklearn.model_selection import train_test_split
import numpy as np

NUM_CLASSES = 5
MUN_FEATURES = 2
SAMPLE_SIZE = 1000

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)  # type: ignore
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)  # type: ignore
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_true)) * 100
    return acc

def save_model(model: nn.Module):
    model_path = Path("saved-models")
    model_name = "multi-class-classification_model.pt"

    model_save_path = model_path / model_name

    torch.save(model.state_dict(), model_save_path)

x_blob, y_blob = make_blobs(n_samples=SAMPLE_SIZE, centers=NUM_CLASSES, n_features=MUN_FEATURES, cluster_std=1.5, random_state=42)

x_blob = torch.tensor(x_blob).float()
y_blob = torch.tensor(y_blob).float()

x_blob_train, x_blob_test, y_blob_train, y_blob_test = train_test_split(x_blob, y_blob, test_size=0.2)

# plt.figure(figsize=(10, 7))
# plt.scatter(x_blob[:, 0], x_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu) # type: ignore
# plt.show()

class MultiClassModel(nn.Module):
    def __init__(self, input_features: int, output_features: int, hidden_units: int=128):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_features)
        )
    def forward(self, x):
        return self.linear_layer_stack(x)
    
model = MultiClassModel(2, 5)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

epochs = 10000
y_pred = model(x_blob_train)

for epoch in range(epochs):
    model.train()
    y_logits = model(x_blob_train)
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)
    acc = accuracy_fn(y_blob_train, y_preds)
    loss = loss_fn(y_logits, y_blob_train.long())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        test_logits = model(x_blob_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits, y_blob_test.long())
        test_acc = accuracy_fn(y_blob_test, test_preds)
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch} | Loss: {loss.item():.4f} | Acc: {acc:.2f}% | Test Loss: {test_loss.item():.4f} | Test Acc: {test_acc:.2f}%")
      
save_model(model)  
# --- UNCOMMENT TO USE SAVED MODEL ---

# saved_model = MultiClassModel(2, 5)
# saved_model.load_state_dict(torch.load("./saved-models/multi-class-classification_model.pt"))

# y_pred = saved_model(x_blob_train)

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Predicted")
# plot_decision_boundary(saved_model, x_blob_train, y_blob_train)
# plt.show()
