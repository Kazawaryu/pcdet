from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np


def vertices(boxes_preds, boxes_labels):
    box1_x1 = boxes_preds[0] - boxes_preds[2] / 2
    box1_y1 = boxes_preds[1] - boxes_preds[3] / 2
    box1_x2 = boxes_preds[0] + boxes_preds[2] / 2
    box1_y2 = boxes_preds[1] + boxes_preds[3] / 2
    box2_x1 = boxes_labels[0] - boxes_labels[2] / 2
    box2_y1 = boxes_labels[1] - boxes_labels[3] / 2
    box2_x2 = boxes_labels[0] + boxes_labels[2] / 2
    box2_y2 = boxes_labels[1] + boxes_labels[3] / 2
    return box1_x1,box2_x1,box1_y1,box2_y1,box1_x2,box2_x2,box1_y2,box2_y2

def iou2d(label,predict):
    label1 = np.concatenate([label[0:1],label[2:3],label[3:4],label[5:6]])

    pred1 = np.concatenate([predict[0:1],predict[2:3],predict[3:4],predict[5:6]])


    box1_x1,box2_x1,box1_z1,box2_z1,box1_x2,box2_x2,box1_z2,box2_z2 = vertices(pred1,label1)
    x1 = max(box1_x1, box2_x1)
    z1 = max(box1_z1, box2_z1)
    x2 = min(box1_x2, box2_x2)
    z2 = min(box1_z2, box2_z2)


    inter = np.clip((x2 - x1),0,None) * np.clip((z2 - z1), 0,None)

    box1 = abs(label[3] * label[5])
    box2 = abs(predict[3] * predict[5])
    iou = inter/(box1+box2-inter)
    return iou

def iou3d(label,predict):
    label1 = np.concatenate([label[0:2],label[3:5]])

    pred1 = np.concatenate([predict[0:2],predict[3:5]])
    box1_x1,box2_x1,box1_y1,box2_y1,box1_x2,box2_x2,box1_y2,box2_y2 = vertices(pred1,label1)
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)

    label2 = np.concatenate([label[1:3], label[4:6]])
    pred2 = np.concatenate([predict[1:3], predict[4:6]])

    box1_y1,box2_y1,box1_z1,box2_z1,box1_y2,box2_y2,box1_z2,box2_z2 = vertices(pred2, label2)
    y1 = max(box1_y1, box2_y1)
    z1 = max(box1_z1, box2_z1)
    y2 = min(box1_y2, box2_y2)
    z2 = min(box1_z2, box2_z2)


    inter = np.clip((x2 - x1),0,None) * np.clip((y2 - y1), 0,None) * np.clip((z2 - z1), 0,None)

    box1 = abs(label[3] * label[4] * label[5])
    box2 = abs(predict[3] * predict[4] * predict[5])
    iou = inter/(box1+box2-inter)
    return iou


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

def plotingcubes(Z1,Z2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X')
    ax.set_zlabel('Y')
    ax.set_ylabel('Z')
    ax.set_xticks(np.arange(-10,10,5))
    ax.set_zticks(np.arange(-10,10,5))
    ax.set_yticks(np.arange(-10,10,5))
    # ax.view_init(elev=24 , azim=-54)

    verts1 = [[Z1[0], Z1[1], Z1[2], Z1[3]],
              [Z1[4], Z1[5], Z1[6], Z1[7]],
              [Z1[0], Z1[1], Z1[5], Z1[4]],
              [Z1[2], Z1[3], Z1[7], Z1[6]],
              [Z1[1], Z1[2], Z1[6], Z1[5]],
              [Z1[4], Z1[7], Z1[3], Z1[0]]]
    ax.add_collection3d(Poly3DCollection(verts1, facecolors='blue', linewidths=0.1, edgecolors='black', alpha=.25))

    verts2 = [[Z2[0], Z2[1], Z2[2], Z2[3]],
              [Z2[4], Z2[5], Z2[6], Z2[7]],
              [Z2[0], Z2[1], Z2[5], Z2[4]],
              [Z2[2], Z2[3], Z2[7], Z2[6]],
              [Z2[1], Z2[2], Z2[6], Z2[5]],
              [Z2[4], Z2[7], Z2[3], Z2[0]]]
    ax.add_collection3d(Poly3DCollection(verts2, facecolors='green', linewidths=0.1, edgecolors='black', alpha=.25))

    plt.show()

def demo():
    # Box = [x,y,z,w,h,l,R]
    # x,y,z = Center position of the Box
    # w,h,l = Dimension of the Box (width, height, length)
    # R = rotation of the Box (radians)

    #Example 1
    # box1 = [4.430, 1.650, 5.200 ,1.570 ,1.650 ,3.350,1.5]
    # box2 = [4.930, 2.250, 5.200 ,1.570 ,1.650 ,3.350,1.4]

    #Example 2
    box1 = [-1.430, 1.650, 1.200, 1.570, 1.650, 2.350, 0.5]
    box2 = [-1.930, 2.250, 0.200, 1.570, 1.650, 2.350, -0.4]

    # IOU 3D
    print('IoU 3D:', np.round(iou3d(box1,box2),4))

    # IOU 2D Bird-eye-view
    print('\nIoU 2D:', np.round(iou2d(box1,box2),4))

    corners1 = get_3d_box(box1)
    corners2 = get_3d_box(box2)
    plotingcubes(corners1,corners2)


def test_from_io():
    predict_path = './../predict_2/019931.txt'
    gt_path = './../label_2/019931.txt'

    predict_dict = {
        'Car': [],
        'Pedestrian': [],
        'Rider': [],
        'Truck': [],
        'Van': []
    }

    gt_dict = {
        'Car': [],
        'Pedestrian': [],
        'Rider': [],
        'Truck': [],
        'Van': []
    }



    with open(predict_path, 'r') as file:
        predict_lines = file.readlines()
        for line in predict_lines:
            line = line.split(' ')
            lab, _, _, _, _, _, _, _, h, w, l, x, y, z, rot, conf = line
            conf = float(conf)
            box = [float(x), float(y), float(z), float(h), float(w), float(l), float(rot)]
            if conf > 0.5: # Confidence threshold
                predict_dict[lab].append(box)
        

    with open(gt_path, 'r') as file:
        gt_lines = file.readlines()
        for line in gt_lines:
            line = line.split(' ')
            lab, _, _, _, _, _, _, _, h, w, l, x, y, z, rot = line
            box = [float(x), float(y), float(z), float(h), float(w), float(l), float(rot)]
            gt_dict[lab].append(box)


    for key in predict_dict.keys():
        for pred in predict_dict[key]:
            max_overlap_3d = 0
            max_overlap_2d = 0
            for gt in gt_dict[key]:
                overlap_3d = iou3d(pred, gt)
                overlap_2d = iou2d(pred, gt)
                max_overlap_3d = max(max_overlap_3d, overlap_3d)
                max_overlap_2d = max(max_overlap_2d, overlap_2d)

            print(f'Class: {key}, IoU 3D: {max_overlap_3d}, IoU 2D: {max_overlap_2d}')
        

    return



if __name__ == '__main__':
    # test_from_io()
    demo()