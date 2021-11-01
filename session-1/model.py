import torch.nn as nn

class MyModel(nn.Module):

    def __init__(self, n_features, n_hidden, n_outputs):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.act2 = nn.Tanh()
        self.fc3 = nn.Linear(n_hidden, n_outputs)
    
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)
