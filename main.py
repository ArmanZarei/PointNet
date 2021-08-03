import torch
from dataset import PointCloudDataset
from path import Path
from transformers import PointSampler, Normalize, RandomRotation, RandomNoise
from torchvision import transforms
from torch.utils.data import DataLoader
from model import PointNet, pointnet_loss
from utils import train_log


# ------------------------ Variables ------------------------ #
path = Path("ModelNet10")
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

# ------------------------ Dataset ------------------------ #
train_dataset = PointCloudDataset(
  root_dir=path,
  folder='train',
  transform=transforms.Compose([
    PointSampler(1024),
    Normalize(),
    RandomRotation(),
    RandomNoise(),
    transforms.ToTensor(),                  
  ])
)
test_dataset = PointCloudDataset(
  root_dir=path,
  folder='test',
  transform=transforms.Compose([
    PointSampler(1024),
    Normalize(),
    transforms.ToTensor(),                  
  ]) 
)

# ------------------------ Data Loaders ------------------------ #
train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=64)