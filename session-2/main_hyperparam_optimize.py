import torch
from ray import tune

from dataset import MyDataset
from model import MyModel
from utils import accuracy, save_model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_single_epoch(...):
    pass


def eval_single_epoch(...):
    pass


def train_model(config, train_dataset, val_dataset):
    my_model = MyModel(...).to(device)
    for epoch in range(config["epochs"]):
        train_single_epoch(...)
        eval_single_epoch(...)


def test_model(config, model, test_dataset):
    pass


if __name__ == "__main__":

    my_dataset = MyDataset(...)
    train_dataset, val_dataset, test_dataset = ...
    ray.init(configure_logging=False)
    analysis = tune.run(
        train_model,
        metric="val_loss",
        mode="min",
        num_samples=5,
        config={
            "hyperparam_1": tune.uniform(1, 10),
            "hyperparam_2": tune.grid_search(["relu", "tanh"]),
        })

    print("Best hyperparameters found were: ", analysis.best_config)
    print(test_model(...))
