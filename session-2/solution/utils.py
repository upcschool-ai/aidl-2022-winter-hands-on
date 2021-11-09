import torch


def accuracy(labels, outputs):
    preds = outputs.argmax(-1)
    acc = (preds == labels.view_as(preds)).float().detach().numpy().mean()
    return acc
    

def save_model(model, path):
    torch.save(model.state_dict(), path)