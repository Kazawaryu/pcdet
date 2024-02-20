import os
import numpy as np
import pandas as pd
import open3d as o3d

def build_kdtree(pcd):
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    return kdtree


def get_points_in_cuboid(kdtree, cuboid):
    indices = []
    # cuboid is the 8 corners of the 3D box, we need to find the points inside the box
    for point in cuboid:
        [_, idx, _] = kdtree.search_knn_vector_3d(point, 1)
        indices.append(idx)


def get_3d_box(box_size):
    def rot(t):
        c = np.cos(t)
        s = np.sin(t)
        # return np.array([[c,  0,  s],
        #                  [0,  1,  0],
        #                  [-s, 0,  c]])

        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])
        # return np.array([[1, 0, 0],
        #                  [0, c, -s],
        #                  [0, s, c]])
    R = rot(box_size[6]+(3.1416/2))
    w, h, l = box_size[3], box_size[4], box_size[5]  # ?, 6, 6, 6, 1
    z_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    x_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]

    corners_3d = np.dot(R,np.vstack([x_corners,  z_corners, y_corners]))
    corners_3d[0, :] = corners_3d[0, :] + box_size[0]
    corners_3d[1, :] = corners_3d[1, :] + box_size[2]
    corners_3d[2, :] = corners_3d[2, :] + box_size[1]
    corners_3d = np.transpose(corners_3d)

    return corners_3d


def single_test(idx):
    label_path = './../label_2/'+idx+'.txt'
    bin_path = './../velodyne/'+idx+'.bin'

    test_dict = {'Car': [], 'Pedestrian': [], 'Rider': [], 'Truck': [], 'Van': []}

    with open(label_path, 'r') as file:
        label_lines = file.readlines()
        for line in label_lines:
            line = line.split(' ')
            lab, _, _, _, _, _, _, _, h, w, l, x, y, z, rot = line
            box = [float(x), float(y), float(z), float(h), float(w), float(l), float(rot)]
            test_dict[lab].append(box)

    pre_point = np.fromfile(str(bin_path), dtype=np.dtype([
                                       ('x', np.float32),
                                       ('y', np.float32),
                                       ('z', np.float32),
                                       ('intensity', np.float32),
                                   ]) ,count=-1)

    pcd = np.array([list(elem) for elem in pre_point])

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:,:3])

    kdtree = build_kdtree(pcd_o3d)

    for lab in test_dict:
        for box in test_dict[lab]:
            corners = get_3d_box(box)
            indices = get_points_in_cuboid(kdtree, corners)
            print(indices)

    return

if __name__ == '__main__':
    single_test('428396')