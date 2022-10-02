from pathlib import Path
from turtle import circle
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def save_model(model: nn.Module):
    model_path = Path("saved-models")
    model_name = "classification_model.pt"

    model_save_path = model_path / model_name

    torch.save(model.state_dict(), model_save_path)


n_samples = 10000

# x is vector/ point in 2D space
# y is label

x, y = make_circles(n_samples=n_samples, noise=0.03, random_state=42)

circles = pd.DataFrame({"x": x[:, 0],
                        "y": x[:, 1],
                        "label": y})
# print(circles)

# plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.RdYlBu) # type: ignore
# plt.show()

X = torch.tensor(x).float()
Y = torch.tensor(y).float()

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)


class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(2, 100)
        self.layer_2 = nn.Linear(100, 100)
        self.layer_3 = nn.Linear(100, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))


model = CircleModel()

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_true)) * 100
    return acc

def plot_predictions(
    train_data, train_labels, test_data, test_labels, predictions=None
):
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})

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
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
epochs = 100000
for epoch in range(epochs):
    model.train()

    y_logits = model(x_train).squeeze()
    y_pred = torch.sigmoid(y_logits).round()

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_train, y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(x_test).squeeze()
        test_pred = torch.sigmoid(test_logits).round()
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_pred)

    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item():.5f}, Acc: {acc:.2f}%, Test Loss: {test_loss.item():.5f}, Test Acc: {test_acc:.2f}%")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Training data")
plot_decision_boundary(model, x_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Testing data")
plot_decision_boundary(model, x_test, y_test)
plt.show()