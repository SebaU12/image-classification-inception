import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from utils.visual_utils import save_img
from utils.train_utils import train_model 
from pytorch_model_summary import summary
from model import Inception_ResNet_v2 

transform = transforms.Compose(
    [
    transforms.Resize((299, 299)),
    transforms.ToTensor()
    ])

torch.manual_seed(20)
batch_size = 16 
num_epochs = 15

train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"CUDA Available: {device}")

def show_description(data_loader):
    imgs, labels = next(iter(data_loader))
    image = imgs[0]  
    label = labels[0]  
    print(f"Label: {label}")
    print(f"Min value: {image.min().item()}")
    print(f"Max value: {image.max().item()}")
    print(f"Size values: {image.size()}")
    save_img(image, label)


def main():
    show_description(train_loader)
    summary(Inception_ResNet_v2(3), torch.zeros((batch_size, 3, 299, 299)), show_input=True, print_summary=True)
    model = Inception_ResNet_v2(3).to(device)
    torch.cuda.empty_cache()
    train_model(model, train_loader, test_loader, num_epochs, device)

if __name__ == '__main__':
    main()
