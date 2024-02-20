import kitti_common as kitti
import os
from eval import get_official_eval_result, get_coco_eval_result
def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]
det_path = "/home/jiazx_ug/OpenPCDet/output/cfgs/kitti_models/pointpillar_copy/default/eval/epoch_159/val/default/final_result/data"


for file_name in os.listdir(det_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(det_path, file_name)
        with open(file_path, 'r') as f:
            lines = f.readlines()
        with open(file_path, 'w') as f:
            for line in lines:
                elements = line.strip().split(' ')
                last_element = float(elements[-1])
                if last_element >= 0.3:
                    f.write(line)


dt_annos = kitti.get_label_annos(det_path)
gt_path = "/home/jiazx_ug/OpenPCDet/data/kitti/training/label_2"
gt_split_file = "/home/jiazx_ug/OpenPCDet/data/kitti/ImageSets/val.txt" 
val_image_ids = _read_imageset_file(gt_split_file)
gt_annos = kitti.get_label_annos(gt_path, val_image_ids)
print(get_official_eval_result(gt_annos, dt_annos, 0))