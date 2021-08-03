import numpy as np
import math
import random


class PointSampler(object):
  def __init__(self, output_size):
    self.output_size = output_size
  
  def triangle_area(self, p1, p2, p3):
    a = np.linalg.norm(p1 - p2)
    b = np.linalg.norm(p2 - p3)
    c = np.linalg.norm(p3 - p1)
    s = (a + b + c)/2
    return math.sqrt(max(s*(s-a)*(s-b)*(s-c), 0)) 
  
  def sample_point_in_triangle(self, p1, p2, p3):
    s, t = sorted([random.random(), random.random()])
    return [s * p1[i] + (t-s) * p2[i] + (1-t) * p3[i] for i in range(3)]
  
  def __call__(self, mesh):
    vertices, faces = mesh

    areas = np.array([self.triangle_area(vertices[f[0]], vertices[f[1]], vertices[f[2]]) for f in faces])
    sampled_faces = random.choices(faces, weights=areas, k=self.output_size)
    pointcloud = np.array([self.sample_point_in_triangle(vertices[f[0]], vertices[f[1]], vertices[f[2]]) for f in sampled_faces])

    return pointcloud


class Normalize(object):
  def __call__(self, pointcloud):
    norm_pointcloud = pointcloud - pointcloud.mean(axis=0)
    norm_pointcloud /= np.linalg.norm(norm_pointcloud, axis=1).max()

    return norm_pointcloud


class RandomRotation(object):
  def __call__(self, pointcloud):
    theta = random.random() * 2 * np.pi # Rotation angle
    rotation_mat = np.array([
      [np.cos(theta), -np.sin(theta), 0],
      [np.sin(theta), np.cos(theta), 0],
      [0, 0, 1]
    ])
    rotated_pointcloud = rotation_mat.dot(pointcloud.T).T

    return rotated_pointcloud


class RandomNoise(object):
  def __call__(self, pointcloud):
    noise = np.random.normal(0, 0.02, pointcloud.shape)
    noisy_pointcloud = pointcloud + noise
    
    return noisy_pointcloud