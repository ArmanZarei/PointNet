from torch.utils.data import Dataset
import os
from path import Path
from utils import read_off


class PointCloudDataset(Dataset):
  def __init__(self, root_dir, folder, transform):
    self.transforms = transform
    
    folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
    self.classes = {folder: i for i, folder in enumerate(folders)}
    
    self.files = []
    for category in self.classes.keys():
      dir_to_search = root_dir/Path(category)/folder
      for file in os.listdir(dir_to_search):
        if file.endswith('.off'):
          self.files.append({
              'path': dir_to_search/file,
              'category': category
          })
  
  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    with open(self.files[idx]['path'], 'r') as f:
      pointcloud = self.transforms(read_off(f))

    return pointcloud, self.classes[self.files[idx]['category']]