import sys
from train import train
from predict import predict

if sys.argv[1] == "train":
    train()

if sys.argv[1] == "predict":
    input_features = [float(feat) for feat in sys.argv[2].split(",")]
    predict(input_features)
