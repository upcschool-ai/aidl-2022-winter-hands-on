import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import MyDataset
from model import MyModel

parser = argparse.ArgumentParser()
parser.add_argument("--n_samples", help="amount of samples to train with", type=int, default=1000)
parser.add_argument("--n_features", help="amount of features per sample", type=int, default=20)
parser.add_argument("--n_hidden", help="amount of hidden neurons", type=int, default=64)
parser.add_argument("--n_outputs", help="amount of outputs", type=int, default=10)
parser.add_argument("--epochs", help="number of epochs to train", type=int, default=5)
parser.add_argument("--batch_size", help="batch size", type=int, default=100)
parser.add_argument("--lr", help="learning rate", type=float, default=0.1)
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

my_dataset = MyDataset(args.n_samples, args.n_features, args.n_outputs)
dataloader = DataLoader(my_dataset, batch_size=args.batch_size)

my_model = MyModel(args.n_features, args.n_hidden, args.n_outputs).to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(my_model.parameters(), lr=args.lr)

loss_history = []

for epoch in range(args.epochs):
    print(f"Epoch {epoch+1}/{args.epochs}")
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_ = my_model(x)
        loss = criterion(y_, y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

plt.plot(loss_history)
plt.title("Training loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.show()
