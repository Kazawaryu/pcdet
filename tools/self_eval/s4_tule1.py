import os
import numpy as np
import pandas as pd
import pickle

import iou3d

CONF_THRESH = 0 # confidence threshold
LIMIT_FILTER = {'Car':15, 'Pedestrian':5, 'Rider':5, 'Truck':15, 'Van':15}

def get_aps(pred, gt, ap_thresh,  ftype , conf_thresh=CONF_THRESH):
    aps = {
        'tp': {'Car': 0,'Pedestrian': 0,'Rider': 0,'Truck': 0,'Van': 0},
        'fp': {'Car': 0, 'Pedestrian': 0, 'Rider': 0, 'Truck': 0, 'Van': 0},
        'fn': { 'Car': 0, 'Pedestrian': 0, 'Rider': 0, 'Truck': 0, 'Van': 0}
    }

    func = iou3d.iou3d if ftype == '3d' else iou3d.iou2d

    for label_keys in pred.keys():
        pred_bbox = pred[label_keys]
        gt_bbox = gt[label_keys]

        max_iou = 0
        deted = [False for i in range(len(gt_bbox))]
        for i in range(len(pred_bbox)):
            for j in range(len(gt_bbox)):
                if not deted[j]:
                    max_iou = max(max_iou, func(pred_bbox[i], gt_bbox[j]))
          
            if max_iou > ap_thresh:
                deted[j] = True
                aps['tp'][label_keys] += 1
            else:
                aps['fp'][label_keys] += 1

        aps['fn'][label_keys] = len(deted) - sum(deted)

    return aps

def get_pr(pred, gt, ap_thresh,ftype, conf_thresh):
    aps = get_aps(pred, gt, ap_thresh,ftype, conf_thresh)

    tps = { 'Car': 0, 'Pedestrian': 0, 'Rider': 0, 'Truck': 0, 'Van': 0 }
    fps = { 'Car': 0, 'Pedestrian': 0, 'Rider': 0, 'Truck': 0, 'Van': 0 }
    fns = { 'Car': 0, 'Pedestrian': 0, 'Rider': 0, 'Truck': 0, 'Van': 0 }
    ps = { 'Car': 0, 'Pedestrian': 0, 'Rider': 0, 'Truck': 0, 'Van': 0 }
    rs = { 'Car': 0, 'Pedestrian': 0, 'Rider': 0, 'Truck': 0, 'Van': 0 }

    for label_keys in pred.keys():
        tps[label_keys] = aps['tp'][label_keys]
        fps[label_keys] = aps['fp'][label_keys]
        fns[label_keys] = aps['fn'][label_keys]

    for label_keys in pred.keys():
        if tps[label_keys] + fps[label_keys] != 0:
            ps[label_keys] = tps[label_keys] / (tps[label_keys] + fps[label_keys])
        else:
            ps[label_keys] = 0

        if tps[label_keys] + fns[label_keys] != 0:
            rs[label_keys] = tps[label_keys] / (tps[label_keys] + fns[label_keys])
        else:
            rs[label_keys] = 0

    return ps, rs

def get_map(idx_list, pred_list, gt_list, ap_thresh,ftype ,conf_thresh,kind):
    aps = {'Car': 0, 'Pedestrian': 0, 'Rider': 0, 'Truck': 0, 'Van': 0}
    ps = {'Car': 0, 'Pedestrian': 0, 'Rider': 0, 'Truck': 0, 'Van': 0}
    rs = {'Car': 0, 'Pedestrian': 0, 'Rider': 0, 'Truck': 0, 'Van': 0}

    save_aps = []

    if kind == 'hard':
        save_path = '/home/jiazx_ug/OpenPCDet/tools/self_eval/hard_result.txt'
    else:
        save_path = '/home/jiazx_ug/OpenPCDet/tools/self_eval/normal_result.txt'


    for i in range(len(pred_list)):
        pred = pred_list[i]
        gt = gt_list[i]
        ps, rs = get_pr(pred, gt, ap_thresh, ftype ,conf_thresh)
        save_aps.append(ps['Car']* rs['Car'])
        for label_keys in pred.keys():
            aps[label_keys] += ps[label_keys] * rs[label_keys]

    with open(save_path, 'w') as file:
        for i in range(len(save_aps)):
            file.write(str(idx_list[i]) + ' ' + str(save_aps[i]) + '\n')

    for label_keys in pred.keys():
        aps[label_keys] /= len(pred_list) 
        aps[label_keys] *= 100
    
    return aps


def test_read_data(file_idx):
    pred_list = []
    gt_list = []

    predict_path = './../predict_2/' + str(file_idx) + '.txt'
    gt_path = './../label_2/' + str(file_idx) + '.txt'

    predict_dict = { 'Car': [], 'Pedestrian': [], 'Rider': [], 'Truck': [], 'Van': [] }
    gt_dict = { 'Car': [], 'Pedestrian': [], 'Rider': [], 'Truck': [], 'Van': [] }

    with open(predict_path, 'r') as file:
        predict_lines = file.readlines()
        for line in predict_lines:
            line = line.split(' ')
            lab, _, _, _, _, _, _, _, h, w, l, x, y, z, rot, conf = line
            conf = float(conf)
            box = [float(x), float(y), float(z), float(h), float(w), float(l), float(rot)]
            if conf > CONF_THRESH:
                predict_dict[lab].append(box)

    with open(gt_path, 'r') as file:
        gt_lines = file.readlines()
        for line in gt_lines:
            line = line.split(' ')
            lab, _, _, _, _, _, _, _, h, w, l, x, y, z, rot = line
            box = [float(x), float(y), float(z), float(h), float(w), float(l), float(rot)]
            gt_dict[lab].append(box)

    pred_list.append(predict_dict)
    gt_list.append(gt_dict)

    mAP = get_map(pred_list, gt_list, 0.5,'3d', CONF_THRESH, 'hard')

    return mAP

def test_all_data(ap_thresh, dist_thresh, ftype, kind):
    # predict_dir = './../predict_2/'
    # gt_dir = './../label_2/'
    # pkl_path = './../kitti_infos_val.pkl'

    predict_dir = '/home/jiazx_ug/OpenPCDet/output/cfgs/kitti_models/pointpillar_copy/default/eval/epoch_240/val/default/final_result/data/'
    gt_dir = '/home/jiazx_ug/OpenPCDet/data/kitti/training/label_2/'
    pkl_path = '/home/jiazx_ug/OpenPCDet/data/kitti/kitti_infos_val.pkl'


    # only consider the id of the file in predict_2
    file_idx = os.listdir(predict_dir)
    file_idx = [x.split('.')[0] for x in file_idx]
    file_idx.sort()
    file_coef = []
    with open(pkl_path, 'rb') as file:
        gt_pkl = pickle.load(file)

    pred_list, gt_list, idx_list = [], [], []

    
    for i in range(len(file_idx)):
        # idx = file_idx[i]
        # output gt_pkl idx
        print(gt_pkl[i]['image']['image_idx'])
        idx = str(gt_pkl[i]['image']['image_idx'])
        num_gts = gt_pkl[i]['annos']['num_points_in_gt']
        idx_list.append(idx)
        predict_dict = { 'Car': [], 'Pedestrian': [], 'Rider': [], 'Truck': [], 'Van': [] }
        gt_dict = { 'Car': [], 'Pedestrian': [], 'Rider': [], 'Truck': [], 'Van': [] }

        predict_path = predict_dir + idx + '.txt'
        gt_path = gt_dir + idx + '.txt'

        with open(predict_path, 'r') as file:
            predict_lines = file.readlines()
            for line in predict_lines:
                line = line.split(' ')
                lab, _, _, _, _, _, _, _, h, w, l, x, y, z, rot, conf = line
                conf = float(conf)
                box = [float(x), float(y), float(z), float(h), float(w), float(l), float(rot)]
            
                predict_dict[lab].append(box)
                file_coef.append((idx,lab,conf))
        print(gt_path)

        with open(gt_path, 'r') as file:
            gt_lines = file.readlines()
            for j in range(len(gt_lines)-1):
                if j >= len(num_gts):
                    print('bad performance, break iter')
                    break
                line = gt_lines[j].split(' ')
                num_gt = num_gts[j]
                if kind == 'normal':
                    lab = line[0]
                    if num_gt < LIMIT_FILTER[lab]:
                        continue
                    else:
                        lab, _, _, _, _, _, _, _, h, w, l, x, y, z, rot = line
                        box = [float(x), float(y), float(z), float(h), float(w), float(l), float(rot)]
                        gt_dict[lab].append(box)
                elif kind == 'hard':
                    if num_gt == 0:
                        continue
                    else:
                        lab, _, _, _, _, _, _, _, h, w, l, x, y, z, rot = line
                        box = [float(x), float(y), float(z), float(h), float(w), float(l), float(rot)]
                        dist = np.sqrt(box[0]**2 + box[1]**2 + box[2]**2)
                        if dist < dist_thresh:
                            gt_dict[lab].append(box)

        pred_list.append(predict_dict)
        gt_list.append(gt_dict)

    mAP = get_map(idx_list, pred_list, gt_list, ap_thresh, ftype ,CONF_THRESH,kind)

    return mAP, file_coef
    
def main():
    metric = [70,50,25]
    maps_n, maps_h = [], []
    for i in metric:
        mapsn,coefn = test_all_data(i/100, 100, '3d', 'normal')
        mapsh,coefh = test_all_data(i/100, 100, '3d', 'hard')
        maps_n.append(mapsn)
        maps_h.append(mapsh)
        # maps_n.append(test_all_data(i/100, 100, '3d', 'normal'))
        # maps_h.append(test_all_data(i/100, 100, '3d', 'hard'))


    index3d = ['mAP@' + str(i) + '_normal'for i in metric]
    index2d = ['mAP@' + str(i) + '_hard' for i in metric]
    ap3d_df = pd.DataFrame(maps_n, index=index3d)
    ap2d_df = pd.DataFrame(maps_h, index=index2d)

    # combine 3d and 2d
    ap_df = pd.concat([ap3d_df, ap2d_df])
    print(ap_df.T)


if __name__ == '__main__':
    main()

