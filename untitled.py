import torch

device = torch.device('cpu')
model = torch.load('model/model_.pth', map_location=device)
torch.save(model, 'model.pth')