from model import PointNet
import torch
from dataset import PointCloudDataset
from path import Path
from torchvision import transforms
from transformers import PointSampler, Normalize
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from utils import confusion_matrix_fig_save


def test(saved_model_path='save.pth'):
    test_dataset = PointCloudDataset(
        root_dir=Path("ModelNet10"),
        folder='test',
        transform=transforms.Compose([
            PointSampler(1024),
            Normalize(),
            transforms.ToTensor(),                  
        ]) 
    )
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=64)

    model = PointNet()
    model.load_state_dict(torch.load(saved_model_path))

    predicts_arr, labels_arr = [], []
    model.eval()
    with torch.no_grad():
        for input, labels in test_dataloader:
            input, labels = input.squeeze().float(), labels
            outputs, _, _ = model(input.transpose(1, 2))
            predicts = torch.argmax(outputs, dim=1)
            labels_arr += labels
            predicts_arr += predicts
    
    return labels_arr, predicts_arr


def incorrect_predicts_fig_save(num_images=(4, 4), output_path='images/incorrect_predictions.png'):
    model = PointNet()
    model.load_state_dict(torch.load('save.pth'))

    test_dataset = PointCloudDataset(
        root_dir=Path("ModelNet10"),
        folder='test',
        transform=transforms.Compose([
            PointSampler(1024),
            Normalize(),
            transforms.ToTensor(),                  
        ]) 
    )
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=20, shuffle=True)\

    class_mapping = {v: k for k, v in test_dataset.classes.items()}

    fig = plt.figure(figsize=(15, 15))
    cnt_found = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.squeeze().float(), labels
            outputs, _, _ = model(inputs.transpose(1, 2))
            predicts = torch.argmax(outputs, dim=1)
            for i in range(20):
                input, label, predict = inputs[i], labels[i], predicts[i]
                if predict != label:
                    ax = fig.add_subplot(num_images[0], num_images[1], cnt_found+1, projection="3d")
                    ax.scatter(input[:, 0], input[:, 1], input[:, 2])
                    ax.set_title(f'Label: {class_mapping[label.item()]} - Predict: {class_mapping[predict.item()]}')
                    ax.set_axis_off()
                    cnt_found += 1
                    if cnt_found == num_images[0]*num_images[1]:
                        fig.savefig(output_path)
                        return
                    


if __name__ == '__main__':
    labels_arr, predicts_arr = test()
    cm = confusion_matrix(labels_arr, predicts_arr)
    confusion_matrix_fig_save(cm)
    print(cm)

    incorrect_predicts_fig_save()