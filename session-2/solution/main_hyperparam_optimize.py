import torch
import pathlib
from torch.utils.data import DataLoader

from dataset import MyDataset
from model import MyModel
from utils import accuracy
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from ray import tune

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_single_epoch(model, train_loader, optimizer):
    model.train()
    accs, losses = [], []
    for x, y in train_loader:
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        y_ = model(x)
        loss = F.cross_entropy(y_, y)
        loss.backward()
        optimizer.step()
        acc = accuracy(y, y_)
        losses.append(loss.item())
        accs.append(acc.item())
    return np.mean(losses), np.mean(accs)


def eval_single_epoch(model, val_loader):
    accs, losses = [], []
    with torch.no_grad():
        model.eval()
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            y_ = model(x)
            loss = F.cross_entropy(y_, y)
            acc = accuracy(y, y_)
            losses.append(loss.item())
            accs.append(acc.item())
    return np.mean(losses), np.mean(accs)


def train_model(config):
    data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    my_dataset = MyDataset("/Users/txus/code/aidl-2022-winter-hands-on/session-2/solution/data/data/data/", 
                           "/Users/txus/code/aidl-2022-winter-hands-on/session-2/solution/data/chinese_mnist.csv", 
                           transform=data_transforms)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(my_dataset, [10000, 2500, 2500])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    my_model = MyModel(config["h1"], config["h2"], config["h3"], config["h4"]).to(device)

    optimizer = optim.Adam(my_model.parameters(), config["lr"])
    for epoch in range(config["epochs"]):
        loss, acc = train_single_epoch(my_model, train_loader, optimizer)
        print(f"Train Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
        val_loss, val_acc = eval_single_epoch(my_model, val_loader)
        print(f"Eval Epoch {epoch} loss={val_loss:.2f} acc={val_acc:.2f}")
        tune.report(val_loss=val_loss)
    
    test_loss, test_acc = eval_single_epoch(my_model, test_loader)
    print(f"Test loss={test_loss:.2f} acc={test_acc:.2f}")
    tune.report(test_loss=test_loss)

    return my_model


if __name__ == "__main__":

    analysis = tune.run(
        train_model,
        metric="val_loss",
        mode="min",
        num_samples=5,
        config = {
            "lr": tune.loguniform(1e-4, 1e-2),
            "batch_size": 64,
            "epochs": 5,
            "h1": tune.randint(20, 40),
            "h2": tune.randint(40, 80),
            "h3": tune.randint(100, 140),
            "h4": tune.randint(100, 140),
        })

    print("Best hyperparameters found were: ", analysis.best_config)
