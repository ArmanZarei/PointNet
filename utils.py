import plotly.graph_objects as go
import numpy as np

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