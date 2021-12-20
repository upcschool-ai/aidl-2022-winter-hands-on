from model import RegressionModel
import torch

@torch.no_grad()
def predict(input_features):
    # load the checkpoint from the correct path
    checkpoint = torch.load("/checkpoints/checkpoint.pt")

    # Instantiate the model
    model = RegressionModel(checkpoint["input_size"], checkpoint["hidden_size"])
    model.load_state_dict(checkpoint["model_state_dict"])

    # Input features is a list of floats. We have to convert it to tensor of the correct shape
    x = torch.tensor(input_features).unsqueeze(0)

    # Now we have to do the same normalization we did when training:
    x = (x - checkpoint["x_mean"])/checkpoint["x_std"]

    # We get the output of the model and we print it
    output = model(x)

    # We have to revert the target normalization that we did when training:
    output = (output * checkpoint["y_std"]) + checkpoint["y_mean"]
    print(f"The predicted price is: ${output.item()*1000:.2f}")
