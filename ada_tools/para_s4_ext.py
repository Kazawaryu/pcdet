import argparse
import glob
from pathlib import Path
import open3d
import torch
import kornia
import numpy as np
import torch
import time
import demo as demo

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.models.backbones_3d.vfe import pillar_vfe as VFE
from pcdet.models.backbones_2d.map_to_bev import pointpillar_scatter as Scatter

from sklearn import svm

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


class exp:
    def __init__(self) -> None:
        self.ckpt_path = '../self_ckpts/pointpillar_7728.pth'
        self.cfg_path = './cfgs/kitti_models/pointpillar.yaml'
        self.pcd_path = '/home/ghosnp/dataset/mini_kitti/velodyne/training/velodyne/000011.bin'
        self.txt_path = '/home/ghosnp/dataset/mini_kitti/label_2/training/label_2/000011.txt'
        self.pcd_dir = '/home/ghosnp/dataset/mini_kitti/velodyne/training/velodyne/'
        self.txt_dir = '/home/ghosnp/dataset/mini_kitti/label_2/training/label_2/'

        self.voxel_size = torch.tensor([0.16, 0.16, 4])
        self.pc_range = torch.tensor([-69.12, -39.68, -3., 69.12, 39.68, 1.])
        self.num_point_features = 4
        self.POINT_CLOUD_RANGE = np.array([-69.12, -39.68, -3, 69.12, 39.68, 1], dtype=np.float32)
        self.range_tool = isPointInQuadrangle()

    def trans_data_2_tensor(self,batch_dict):
        for key, val in batch_dict.items():
            if key == 'camera_imgs':
                batch_dict[key] = val.cuda()
            elif not isinstance(val, np.ndarray):
                continue
            elif key in ['frame_id', 'metadata', 'calib', 'image_paths','ori_shape','img_process_infos']:
                continue
            elif key in ['images']:
                batch_dict[key] = kornia.image_to_tensor(val).float()
            elif key in ['image_shape']:
                batch_dict[key] = torch.from_numpy(val).int()
            else:
                batch_dict[key] = torch.from_numpy(val).float()
        
        return

    def load_model_paras(self):
        in_channels = 10
        out_channels = 64
        use_norm = True
        last_layer = True
        grid_size = [(self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0], (self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1], 1]
        grid_size = torch.tensor(grid_size).int()

        self.logger = common_utils.create_logger()
        cfg_from_yaml_file(self.cfg_path, cfg)

        init_dataset = demo.DemoDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path(self.pcd_path), ext='.bin', logger=self.logger)

        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=init_dataset)
        model.load_params_from_file(filename=self.ckpt_path, logger=self.logger, to_cpu=True)

        self.pillar_vfe = VFE.PillarVFE(model_cfg=model.model_cfg.VFE, num_point_features=self.num_point_features, voxel_size=self.voxel_size, point_cloud_range=self.pc_range)
        self.pillar_scatter = Scatter.PointPillarScatter(model_cfg=model.model_cfg.MAP_TO_BEV, grid_size=grid_size)
        
        return model
    
    def vfe_process(self, data_dict):
        self.pillar_vfe.forward(data_dict)
        self.pillar_scatter.forward(data_dict)

        return  data_dict['spatial_features']

    def fit_hyper_plane(self, model):
        # get vfe_pfn_weight from model_cfg
        vfe_pfn_weight_gpu = model.vfe.pfn_layers[0].linear.weight
        vfe_pfn_weight = vfe_pfn_weight_gpu.detach().cpu().numpy()

        self.clf = svm.OneClassSVM(nu=0.1, kernel = "rbf", gamma=0.05)
        self.clf.fit(vfe_pfn_weight.T)

        return

    def load_data(self, pcd_path):
        single_dataset = demo.DemoDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path(pcd_path), ext='.bin', logger=self.logger)
        
        with torch.no_grad():
            for idx, data_dict in enumerate(single_dataset):
                # only one in fact
                data_dict = single_dataset.collate_batch([data_dict])
                self.trans_data_2_tensor(data_dict)

        return data_dict
    
    def get_feature_idx(self, txt_path):
        corners_bevs = []

        with open(txt_path, 'r') as f:
            txt = f.readlines()

        for line in txt:
            line = line.split()
            lab, x, y, z,h, w, l, rot = line[0], line[11], line[12], line[13], line[8], line[9], line[10], line[14]
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
                corners_bevs.append(corners_3d[:4, [0, 1]])

        corners_bevs = np.array(corners_bevs)

        # corners_bevs add range offset
        corners_bevs[:, :, 0] += self.POINT_CLOUD_RANGE[3]
        corners_bevs[:, :, 1] += self.POINT_CLOUD_RANGE[4]
        corners_bevs /= 0.16

        feature_idx = []
        for corners in corners_bevs:
            max_x, min_x, max_y, min_y = np.max(corners[:,0]), np.min(corners[:,0]), np.max(corners[:,1]), np.min(corners[:,1])
            max_x, min_x, max_y, min_y = int(max_x), int(min_x), int(max_y), int(min_y)
            # get x,y with (min_x < x < max_x and min_y < y < max_y)
            if max_x < 864 and max_y < 496:
                x = np.arange(min_x, max_x)
                y = np.arange(min_y, max_y)
                for i in x:
                    for j in y:
                        
                        a, b, c, d = self.range_tool.compute_para(i, j, corners[0][0], corners[0][1], corners[1][0], corners[1][1], corners[2][0], corners[2][1], corners[3][0], corners[3][1])
                        res = self.range_tool.is_in_rect(a, b, c, d)
                        if res:
                            feature_idx.append([i, j])
            
        feature_idx = np.array(feature_idx)

        return feature_idx
        
    def iter_get_s4(self):
        pcd_dir, txt_dir = self.pcd_dir, self.txt_dir
        model = self.load_model_paras()
        self.fit_hyper_plane(model)
        dist_dict = {}
        mean_dist = []
        tick = time.time()
        print('start')
        for pcd_path in glob.glob(pcd_dir + '*.bin'):
            txt_path = txt_dir + pcd_path.split('/')[-1].split('.')[0] + '.txt'
            print('pcd_path: ', pcd_path)
            iter_data_dict = self.load_data(pcd_path)
            all_features = self.vfe_process(iter_data_dict)
            feature_idx = self.get_feature_idx(txt_path)
            feature_idx_gpu = torch.from_numpy(feature_idx).cuda()
            spatial_features_gpu = all_features[0]

            vaild_features_gpu = spatial_features_gpu[:, feature_idx_gpu[:,1], feature_idx_gpu[:,0]]
            vaild_feature = vaild_features_gpu.detach().numpy().T

            dists = np.abs(self.clf.decision_function(vaild_feature))
            dist_dict[pcd_path] = dists
            mean_dist.append(np.mean(dists))
            print('time: ', time.time() - tick, 's')
            tick = time.time()


        return dist_dict, mean_dist
    
if __name__ == '__main__':
    s4_exp = exp()
    dist_dict, mean_dist = s4_exp.iter_get_s4()
    print(mean_dist)
    

        

