import os
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from model import RegressionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(dataloader, model, optimizer, criterion):

    # Train the model
    train_loss = 0
    for X, y in dataloader:
        optimizer...
        X, y = X.to(device), y.to(device)
        y_ = ...
        loss = ...
        train_loss += loss.item() * len(y)
        loss...
        optimizer...

    return train_loss / len(dataloader.dataset)


def test_epoch(dataloader: DataLoader, model, criterion):
    test_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            y_ = ...
            loss = ...
            test_loss += loss.item() * len(y)

    return test_loss / len(dataloader.dataset)


def load_data():
    df = pd.read_csv("/data/housing.csv")
    train_df, test_df = train_test_split(df, test_size=0.2)
    train_X, train_y = train_df.drop(["ID", "MEDV"], axis=1), train_df["MEDV"]
    test_X, test_y = test_df.drop(["ID", "MEDV"], axis=1), test_df["MEDV"]
    train_X, train_y = train_X.to_numpy(), train_y.to_numpy()
    test_X, test_y = test_X.to_numpy(), test_y.to_numpy()
    return train_X, train_y, test_X, test_y


def train():

    # Hyperparameters
    BATCH_SIZE = 16
    N_EPOCHS = 10
    HIDDEN_SIZE = 64

    # This function reads the CSV using pandas and splits it into train and test using sklearn.
    # Then, it converts the data into numpy tensors. Feel free to take a look at it if you are curious
    train_X, train_y, test_X, test_y = load_data()

    # Now you should normalize your data (without Data Leakage) and wrap it in a Dataset

    # Start by converting the numpy tensors to torch tensors
    train_X, train_y = ...
    test_X, test_y = ...

    # Compute the mean and std in the correct dimension
    x_mean = ...
    x_std = ...
    y_mean = ...
    y_std = ...

    # Apply it to our data ((data - mean)/std)
    train_X = ...
    train_y = ...

    test_X = ...
    test_y = ...

    # Instantiate the datasets
    train_dataset = ...
    test_dataset = ...

    # Instantiate data loaders
    train_loader = ...
    test_loader = ...

    input_size = train_X.shape[1]

    # Load the model
    model = ...
        
    # You should use a loss function appropriate for regression
    criterion = ...

    # Setup optimizer. SGD with lr=0.1 will work
    optimizer = ...

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        train_loss = train_epoch(train_loader, model, optimizer, criterion)
        test_loss = test_epoch(test_loader, model, criterion)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        print(f"Epoch: {epoch + 1},  | time in {mins} minutes, {secs} seconds")
        print(f'\tLoss: {train_loss:.4f}(train)')
        print(f'\tLoss: {test_loss:.4f}(test)')

    # Now save the artifacts of the training
    # Do not change this path (unless debugging). You should mount a Docker volume to it
    savedir = "/checkpoints/checkpoint.pt"
    print(f"Saving checkpoint to {savedir}...")
    # We can save everything we will need later in the checkpoint.
    # Here, we could save a feature transformer if we had used one
    checkpoint = {
        "model_state_dict": ...,
        "optimizer_state_dict": ...,
        "input_size": ...,
        "hidden_size": ...,
        "x_mean": ...,
        "y_mean": ...,
        "x_std": ...,
        "y_std": ...,
    }
    torch.save(...)

if __name__ == "__main__":
    train()
