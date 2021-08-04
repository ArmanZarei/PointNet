import plotly.graph_objects as go
import numpy as np
import re
from matplotlib import pyplot as plt
import seaborn as sns


def plot_pointcloud(pointcloud):
  fig = go.Figure(data=[go.Scatter3d(
  x=pointcloud[:, 0],
  y=pointcloud[:, 1],
  z=pointcloud[:, 2],
  mode='markers',
  marker=dict(
    size=2,
    color='#34495e', 
    colorscale='Viridis',
    opacity=0.8
  ))])
  fig.show()

def read_off(file):
  if 'OFF' != file.readline().strip():
    raise('Not a valid OFF header')
  n_vertices, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
  vertices = np.array([[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_vertices)])
  faces = np.array([[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)])
  return vertices, faces

def train_log(txt, delete_prev=False, file_path="TrainLog.txt"):
  with open(file_path, 'w' if delete_prev else 'a') as f:
    f.write(txt + "\n")

def training_process_plot_save(train_loss_arr, test_loss_arr, train_accuracy_arr, test_accuracy_arr):
  plt.figure(figsize=(20, 8))
  plt.subplot(1, 2, 1).set_title("Loss / Epoch")
  plt.plot(train_loss_arr, label='Train')
  plt.plot(test_loss_arr, label='Validation')
  plt.legend()
  plt.subplot(1, 2, 2).set_title("Accuracy / Epoch")
  plt.plot(train_accuracy_arr, label='Train')
  plt.plot(test_accuracy_arr, label='Validation')
  plt.legend()
  plt.savefig('images/training.png')


def save_training_process():
  train_loss_arr, test_loss_arr = [], []
  train_accuracy_arr, test_accuracy_arr = [], []
  patt = re.compile("Epoch:\s+\d+ ->\s+Train Loss: (\d+\.\d+)\s+Test Loss: (\d+\.\d+) \| Train Accuracy: (\d\.\d+)\s+Test Accuracy: (\d\.\d+)")
  with open('TrainLog.txt', 'r') as f:
    for line in f.readlines()[2:]:
      m = patt.search(line)
      train_loss_arr.append(float(m.group(1)))
      test_loss_arr.append(float(m.group(2)))
      train_accuracy_arr.append(float(m.group(3)))
      test_accuracy_arr.append(float(m.group(4)))
  training_process_plot_save(train_loss_arr, test_loss_arr, train_accuracy_arr, test_accuracy_arr)

def confusion_matrix_fig_save(confusion_matrix, output_path='images/confusion_matrix.png'):
  classes = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
  plt.figure(figsize=(12, 10))
  fig = sns.heatmap(
    confusion_matrix,
    xticklabels=classes,
    yticklabels=classes,
    annot=True,
    cmap=plt.cm.Blues
  )
  plt.xticks(rotation=45)
  fig.figure.savefig(output_path)