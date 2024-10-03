import torch
from model import Inception_ResNet_v2
print(torch.__version__)  # Verifica la versión de PyTorch
print(torch.cuda.is_available())  # Verifica si CUDA está disponible


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Inception_ResNet_v2(3).to(device)
torch.optim.Adam(model.parameters(), 0.001)
