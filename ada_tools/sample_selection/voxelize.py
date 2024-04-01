import sys
import numpy as np
import torch
import torch.nn as nn
import cumm.tensorview as tv
import spconv.pytorch as spconv
from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
from functools import partial

class input_backbone(nn.Module):
    def __init__(self, input_channels, grid_size):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key="subm0").to(0),
            norm_fn(16).to(0),
            nn.ReLU()
        )

    def forward(self, voxel_features, voxel_coords):
        sp_tensor = spconv.SparseConvTensor(
            features = voxel_features,
            indices = voxel_coords,
            spatial_shape = self.sparse_shape,
            batch_size = 1
        )

        x = self.conv_input(sp_tensor)
        return x
    
class isPointInQuadrangle(object):
    def __int__(self):
        self.__isInQuadrangleFlag = False

    def cross_product(self, xp, yp, x1, y1, x2, y2):
        return (x2 - x1) * (yp - y1)-(y2 - y1) * (xp - x1)

    def compute_para(self, xp, yp, xa, ya, xb, yb, xc, yc, xd, yd):
        cross_product_ab = isPointInQuadrangle().cross_product(xp, yp, xa, ya, xb, yb)
        cross_product_bc = isPointInQuadrangle().cross_product(xp, yp, xb, yb, xc, yc)
        cross_product_cd = isPointInQuadrangle().cross_product(xp, yp, xc, yc, xd, yd)
        cross_product_da = isPointInQuadrangle().cross_product(xp, yp, xd, yd, xa, ya)
        return cross_product_ab,cross_product_bc,cross_product_cd,cross_product_da

    def is_in_rect(self, aa, bb, cc, dd):
        if (aa > 0 and bb > 0 and cc > 0 and dd > 0) or (aa < 0 and bb < 0 and cc < 0 and dd < 0):
            self.__isInQuadrangleFlag= True
        else:
            self.__isInQuadrangleFlag = False

        return self.__isInQuadrangleFlag

def create_voxel(points, vsize_xyz, coors_range_xyz,num_point_features ,max_num_points_per_voxel, max_num_voxels):
    voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )
    
    tv_voxels, tv_voxel_indices, tv_num_points = voxel_generator.point_to_voxel(tv.from_numpy(points))
    voxels = tv_voxels.numpy()
    voxel_indices = tv_voxel_indices.numpy()
    num_points = tv_num_points.numpy()

    return voxels, voxel_indices, num_points

def meanVFE(voxel_features, voxel_num_points):
    points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False) # Use sum() instead of _sum()
    normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
    points_mean = points_mean / normalizer # sum / count for each voxel
    return points_mean.contiguous()

def get_grid_size(coors_range_xyz, vsize_xyz):
    grid_size = (coors_range_xyz[3:6] - coors_range_xyz[0:3]) / np.array(vsize_xyz)
    grid_size = np.round(grid_size).astype(np.int64)

    return grid_size

def get_input_feature(voxels, num_points, voxel_indices, backbone):
    torch_voxels = torch.from_numpy(voxels)
    torch_num_points = torch.from_numpy(num_points)
    torch_indices = torch.from_numpy(voxel_indices)

    mean_vfe = meanVFE(torch_voxels, torch_num_points)
    coordinates = torch.cat([torch.zeros_like(torch_indices[:, :1]), torch_indices], dim=1)
    for i in range(len(coordinates)):
        coordinates[i, 0] = i

    torch_voxel_features = backbone(mean_vfe.to(0), coordinates.to(0))
    input_features = torch_voxel_features.features.cpu().detach().numpy()
    input_indices = torch_voxel_features.indices.cpu().detach().numpy()
    return input_features, input_indices

def indices_to_3d(indices, grid_size):
    indices_3d = np.zeros(grid_size, dtype=np.int32)
    indices_3d[indices[:,3],indices[:,2],indices[:,1]] = indices[:,0]

    return indices_3d

def corners_to_voxelaxis(corners_3Ds, coors_range_xyz, vsize_xyz):
    corners_voxelaxis = np.array(corners_3Ds)
    corners_voxelaxis = (corners_voxelaxis - coors_range_xyz[0:3] - 0.5 * vsize_xyz) / vsize_xyz
    # corners_voxel = (corners_voxel - coors_range_xyz[0:3]) / vsize_xyz - 0.5
    corners_voxelaxis = corners_voxelaxis.astype(np.int32)

    return corners_voxelaxis

def get_inner_voxel(corners_voxelaxis, checker):
    feature_keys = []
    for num in range(len(corners_voxelaxis)):
        feature_keys.append([])
        corner = corners_voxelaxis[num]
        max_x,min_x,max_y,min_y = np.max(corner[:,0]),np.min(corner[:,0]),np.max(corner[:,1]),np.min(corner[:,1])
        x = np.arange(int(min_x),int(max_x)+1)
        y = np.arange(int(min_y),int(max_y)+1)
        for i in x:
            for j in y:
                a,b,c,d = checker.compute_para(i,j,corner[0][0],corner[0][1],corner[1][0],corner[1][1],corner[2][0],corner[2][1],corner[3][0],corner[3][1])
                if checker.is_in_rect(a,b,c,d):
                    for k in range(corner[0][2],corner[4][2]+1):
                        feature_keys[num].append([i,j,k])

    return feature_keys

def get_inner_voxel_feature(voxel_features, indices_3d, feature_keys, grid_size):
    features = []
    for obj in feature_keys:
        for key in obj:
            if key[0] < 0 or key[0] >= grid_size[0] or key[1] < 0 or key[1] >= grid_size[1] or key[2] < 0 or key[2] >= grid_size[2]:
                continue    
            indice = indices_3d[key[0], key[1], key[2]]
            features.append(voxel_features[indice])
    
    return features


def read_data(path):
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    
    return points

def read_label(path):
    # Attention: here use KITTI label format
    with open(path, 'r') as f:
        labels = f.readlines()

    corners_3Ds = []

    for line in labels:
        line = line.split()
        lab, x, y, z, w, l, h, rot = line[0], line[11], line[12], line[13], line[9], line[10], line[8], line[14]
        h, w, l, x, y, z, rot = map(float, [h, w, l, x, y, z, rot])
        
        if lab != 'DontCare':
            x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
            y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
            z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
            corners_3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)

            # transform the 3d bbox from object coordiante to camera_0 coordinate
            R = np.array([[np.cos(rot), 0, np.sin(rot)],
                        [0, 1, 0],
                        [-np.sin(rot), 0, np.cos(rot)]])
    
            corners_3d = np.dot(R, corners_3d).T + np.array([x, y, z])

            # transform the 3d bbox from camera_0 coordinate to velodyne coordinate
            corners_3d = corners_3d[:, [2, 0, 1]] * np.array([[1, -1, -1]])
            corners_3Ds.append(corners_3d)

    return corners_3Ds


