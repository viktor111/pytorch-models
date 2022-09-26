from inspect import Parameter
import torch
from torch import nn
import matplotlib.pyplot as plt

# workarounds for pylance linter bug
from torch.nn import parameter

TRAIN_SPLIT_VAL = 0.8
EPOCHS = 1000
WEIGHT = 0.7
BIAS = 0.5
START = 0
END = 2
STEP = 0.02


class TrainingResults:
    def __init__(self,
                 train_loss_history,
                 test_loss_history,
                 epoch_counter,
                 model: nn.Module):
        self.train_loss_history = train_loss_history
        self.test_loss_history = test_loss_history
        self.epoch_counter = epoch_counter
        self.model = model


class TrainingData:
    def __init__(self,
                 x_train: torch.Tensor,
                 y_train: torch.Tensor,
                 x_test: torch.Tensor,
                 y_test: torch.Tensor):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


class TrainingOptions:
    def __init__(self, optimizer: torch.optim.Optimizer, loss_fn: nn.Module):
        self.optimizer = optimizer
        self.loss_fn = loss_fn


class TrainingCoreParams:
    def __init__(self, epochs: int, model: nn.Module):
        self.epochs = epochs
        self.model = model


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = parameter.Parameter(torch.zeros(
            1, requires_grad=True, dtype=torch.float32))
        self.bias = parameter.Parameter(torch.zeros(
            1, requires_grad=True, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.weight * x + self.bias
        return result


def plot_predictions(
        train_data,
        train_labels,
        test_data,
        test_labels,
        predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c='b', s=4, label="Training data")
    plt.scatter(test_data, test_labels, c='g',  s=4, label="Test data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s=4, label="Predictions")
    plt.legend(prop={'size': 14})
    plt.show()


def plot_loss_history(
        epoch_counter,
        train_loss_history,
        test_loss_history):
    plt.plot(epoch_counter, train_loss_history, label="Train loss")
    plt.plot(epoch_counter, test_loss_history, label="Test loss")
    plt.title("Loss history")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")


def prepare_data(
        weight: float,
        bias: float,
        start: int,
        end: int,
        step: float) -> TrainingData:
    x = torch.arange(start=start, end=end, step=step).unsqueeze(dim=1)

    y = weight * x + bias

    train_split = int(TRAIN_SPLIT_VAL * len(x))

    x_train = x[:train_split]
    y_train = y[:train_split]

    x_test = x[train_split:]
    y_test = y[train_split:]

    return TrainingData(x_train, y_train, x_test, y_test)


def train(training_core_params: TrainingCoreParams, training_options: TrainingOptions, training_data: TrainingData) -> TrainingResults:
    epoch_counter = []
    train_loss_history = []
    test_loss_history = []
    results = TrainingResults(
        train_loss_history, test_loss_history, epoch_counter, training_core_params.model)
    for epoch in range(training_core_params.epochs):
        training_core_params.model.train()
        training_options.optimizer.zero_grad()
        predictions = training_core_params.model(training_data.x_train)
        loss = training_options.loss_fn(
            predictions, training_data.y_train)
        loss.backward()
        training_options.optimizer.step()

        with torch.no_grad():
            training_core_params.model.eval()
            test_predictions = training_core_params.model(training_data.x_test)
            test_loss = training_options.loss_fn(
                test_predictions, training_data.y_test)

        epoch_counter.append(epoch)
        train_loss_history.append(loss.detach().numpy())
        test_loss_history.append(test_loss.detach().numpy())
        if epoch == EPOCHS - 1:
            results = TrainingResults(
                train_loss_history, test_loss_history, epoch_counter, training_core_params.model)
    return results


# To get better result tweak the parameters
data = prepare_data(WEIGHT, BIAS, START, END, STEP)

training_core_params = TrainingCoreParams(
    epochs=EPOCHS, model=LinearRegressionModel())

training_options = TrainingOptions(
    optimizer=torch.optim.SGD(
        training_core_params.model.parameters(), lr=0.01),
    loss_fn=nn.L1Loss())

training_data = TrainingData(
    data.x_train, data.y_train, data.x_test, data.y_test)

train_results = train(training_core_params, training_options, training_data)

print(f"Original Weight: {WEIGHT} - Bias: {BIAS}")
# type: ignore
print(
    f"Predicted Weight: {train_results.model.weight.item():.3f} - Bias: {train_results.model.bias.item():.3f}") # type: ignore

plot_loss_history(
    train_results.epoch_counter,
    train_results.train_loss_history,
    train_results.test_loss_history)
plot_predictions(
    train_data=data.x_train,
    train_labels=data.y_train,
    test_data=data.x_test,
    test_labels=data.y_test,
    predictions=train_results.model(data.x_test).detach().numpy())
