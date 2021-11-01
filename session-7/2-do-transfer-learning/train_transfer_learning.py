import copy
import os
import time
from urllib.request import urlretrieve
from zipfile import ZipFile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets


def train_model(model, criterion, optimizer, train_loader, val_loader, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        train_epoch(model, train_loader, criterion, optimizer, scheduler)

        epoch_acc = eval_epoch(model, val_loader, criterion)

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


@torch.no_grad()
def eval_epoch(model, val_loader, criterion):
    model.eval()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0
    # Iterate over data.
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_corrects.double() / len(val_loader.dataset)
    print(f"Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    return epoch_acc


def train_epoch(model, train_loader, criterion, optimizer, scheduler):
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0
    # Iterate over data.
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    scheduler.step()
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")


def load_data():
    data_dir = "hymenoptera_data"
    if not os.path.isfile(data_dir):
        zip_name = "hymenoptera_data.zip"
        urlretrieve(
            "https://download.pytorch.org/tutorial/hymenoptera_data.zip",
            "hymenoptera_data.zip",
        )
        with ZipFile(zip_name, "r") as f:
            f.extractall(".")
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), data_transforms["train"])
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), data_transforms["val"])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    return train_loader, val_loader


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = load_data()

    # TODO instantiate a pretrained ResNet18 model from Torchvision
    model = ...
    in_features = model.fc.in_features
    # TODO replace the last layer of the model (model.fc) by a linear layer with the correct input and output size
    ...
    model = model.to(device)

    # TODO Use the correct loss function for classification (not binary). The network doesn't have an activation at the end.
    criterion = ...

    # TODO Use Adam with default lr to train ONLY the new layer of the model
    optimizer = ...
    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_model(
        model, criterion, optimizer, train_loader, val_loader, scheduler, num_epochs=25
    )
