import os

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def test_final_model(gbm, test_X, test_y):
    predictions = gbm.predict(test_X)
    return mean_absolute_error(test_y, predictions)


def load_data():
    df = pd.read_csv("./data/housing.csv")
    train_df, test_df = train_test_split(df, test_size=0.2)
    train_X, train_y = train_df.drop(["ID", "MEDV"], axis=1), train_df["MEDV"]
    test_X, test_y = test_df.drop(["ID", "MEDV"], axis=1), test_df["MEDV"]
    train_X, train_y = train_X.to_numpy(), train_y.to_numpy()
    test_X, test_y = test_X.to_numpy(), test_y.to_numpy()
    return train_X, train_y, test_X, test_y


def train():

    # This function reads the CSV using pandas and splits it into train and test using sklearn.
    # Then, it converts the data into numpy tensors. Feel free to take a look at it if you are curious
    train_X, train_y, test_X, test_y = load_data()

    # TODO instantiate a LightGBMRegressor
    gbm = ...

    # TODO train the gbm with the train dataset
    ...

    print(
        "=" * 40,
        f"Final mean_absolute error = {test_final_model(gbm, test_X, test_y):.2f}",
        "=" * 40,
        sep="\n"
    )

    # Now save the artifacts of the training
    savedir = "checkpoints/model.txt"
    os.makedirs("checkpoints", exist_ok=True)
    print(f"Saving checkpoint to {savedir}...")
    gbm.booster_.save_model(savedir)


if __name__ == "__main__":
    train()
