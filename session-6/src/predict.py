from typing import List

import torch

from model import RegressionModel


@torch.no_grad()
def predict(input_features: List[float]):
    # load the checkpoint from the correct path
    checkpoint = ...

    # Instantiate the model and load the state dict
    model = ...
    model...

    # Input features is a list of floats. We have to convert it to tensor of the correct shape
    x = torch.tensor(input_features).unsqueeze(0)

    # Now we have to do the same normalization we did when training:
    x = ...

    # We get the output of the model and we print it
    output = ...

    # We have to revert the target normalization that we did when training:
    output = ...
    print(f"The predicted price is: ${output.item()*1000:.2f}")
