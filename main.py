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


# ------------------------ Training ------------------------ #
model = PointNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_log(f'Device: {device}\n{"-"*30}', delete_prev=True)

train_loss_arr, test_loss_arr = [], []
train_accuracy_arr, test_accuracy_arr = [], []
for epoch in range(15):
    train_loss, test_loss = .0, .0
    train_acc, test_acc = .0, .0
  
    model.train()
    for input, labels in train_dataloader:
        input, labels = input.to(device).squeeze().float(), labels.to(device)
    
        optimizer.zero_grad()
        outputs, mat_3x3, mat_64x64 = model(input.transpose(1, 2))
    
        loss = pointnet_loss(outputs, labels, mat_3x3, mat_64x64)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += (labels == torch.argmax(outputs, dim=1)).sum().item() / input.shape[0]
  
    model.eval()
    with torch.no_grad():
        for input, labels in test_dataloader:
            input, labels = input.to(device).squeeze().float(), labels.to(device)
            outputs, mat_3x3, mat_64x64 = model(input.transpose(1, 2))
            test_loss += pointnet_loss(outputs, labels, mat_3x3, mat_64x64).item()
            test_acc += (labels == torch.argmax(outputs, dim=1)).sum().item() / input.shape[0]

    train_loss_arr.append(train_loss/len(train_dataloader))
    test_loss_arr.append(test_loss/len(test_dataloader))
    train_accuracy_arr.append(train_acc/len(train_dataloader))
    test_accuracy_arr.append(test_acc/len(test_dataloader))
  
    train_log(f'Epoch: {"{:2d}".format(epoch)} -> \t Train Loss: {"%.10f"%train_loss_arr[-1]} \t Test Loss: {"%.10f"%test_loss_arr[-1]} | Train Accuracy: {"%.4f"%train_accuracy_arr[-1]} \t Test Accuracy: {"%.4f"%test_accuracy_arr[-1]}')

print("OK")