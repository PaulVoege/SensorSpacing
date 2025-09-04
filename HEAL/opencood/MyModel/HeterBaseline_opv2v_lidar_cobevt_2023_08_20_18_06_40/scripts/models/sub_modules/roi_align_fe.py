# Author Yifan Lu <yifan_lu@sjtu.edu.cn>
# use Rotated RoIAlign to extract box features.

from mmcv.ops import RoIAlignRotated
import torch.nn as nn
import torch
import os

class RoIAlignFeatureExtractor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.roi_align_size = args['roi_align_size']
        self.roi_align = RoIAlignRotated(output_size = self.roi_align_size, 
                                         spatial_scale = 1,
                                         clockwise = True)
        self.cav_range = args['cav_range']
        self.enlarge_ratio = args.get("enlarge_ratio", 1)
        self.target_roi_num = args['target_roi_num']
        self.target_save_path = args['target_save_path']
        self.target_file_name = args['target_file_name']
        self.roi_feature_pool = None
        
    def forward(self, data_dict, feature):
        """
        Args:
            data_dict : dict
                The input of the model

                - For intermediate fusion dataset , we use 'object_bbx_center_single' 
                    and 'object_bbx_center_mask_single' for RoI feature extraction

                - For late fusion dataset, we use 'object_bbx_center' 
                    and 'object_bbx_center_mask' for RoI feature extraction
                

            feature : torch.tensor
                The BEV feature after encoder, backbone and aligner (if exists).

        Returns:
            None. 
                We store the RoI feature in self.roi_feature_pool, and save them to file in the end.
        """

        # intermediate fusion
        if "record_len" in data_dict:
            object_bbx_center_single = data_dict['object_bbx_center_single']
            object_bbx_mask_single = data_dict['object_bbx_mask_single']
            assert object_bbx_mask_single.shape[0] == sum(data_dict['record_len'])
        else:
            object_bbx_center_single = data_dict['object_bbx_center']
            object_bbx_mask_single = data_dict['object_bbx_mask']
            assert object_bbx_mask_single.shape[0] == 1

        gt_boxes = [b[m] for b, m in
                    zip(object_bbx_center_single,
                        object_bbx_mask_single.bool())]  # x, y, z, h, w, l, theta
        all_box_num = int(torch.sum(object_bbx_mask_single))
        
        # proposal to rotated roi input, 
        # (batch_index, center_x, center_y, w, h, angle). The angle is in radian.
        roi_input = torch.zeros((all_box_num, 6), device=feature.device)

        H, W = feature.shape[2:]
        
        grid_size_W = (self.cav_range[3] - self.cav_range[0]) / W
        grid_size_H = (self.cav_range[4] - self.cav_range[1]) / H
        box_cnt = 0

        for batch_idx, gt_box in enumerate(gt_boxes):
            # gt_box is [n_boxes, 7], x, y, z, h, w, l, yaw -> (center_x, center_y, w, h)
            roi_input[box_cnt:box_cnt+gt_box.shape[0], 0] = batch_idx
            roi_input[box_cnt:box_cnt+gt_box.shape[0], 1] = (gt_box[:, 0] - self.cav_range[0]) / grid_size_W
            roi_input[box_cnt:box_cnt+gt_box.shape[0], 2] = (gt_box[:, 1] - self.cav_range[1]) / grid_size_H
            roi_input[box_cnt:box_cnt+gt_box.shape[0], 3] = gt_box[:, 5] / grid_size_W * self.enlarge_ratio  # box's l -> W
            roi_input[box_cnt:box_cnt+gt_box.shape[0], 4] = gt_box[:, 4] / grid_size_H * self.enlarge_ratio # box's w -> H
            roi_input[box_cnt:box_cnt+gt_box.shape[0], 5] = gt_box[:, 6] 
            box_cnt += gt_box.shape[0]
        
        # [sum(proposal), C, self.roi_align_size, self.roi_align_size]
        pooled_feature = self.roi_align(feature, roi_input) 
        # [sum(proposal), self.roi_align_size, self.roi_align_size, C]
        pooled_feature = pooled_feature.permute(0,2,3,1).detach().cpu()
        if self.roi_feature_pool is None:
            self.roi_feature_pool = pooled_feature
        self.roi_feature_pool = torch.cat((self.roi_feature_pool, pooled_feature), dim=0)
        print(f"collectd {len(self.roi_feature_pool)} / {self.target_roi_num} roi features...")


        if len(self.roi_feature_pool) > self.target_roi_num:
            print("Collect enough roi features!")
            print("We will save them to file and quit.")
            if not os.path.exists(self.target_save_path):
                os.mkdir(self.target_save_path)
            torch.save(self.roi_feature_pool, os.path.join(self.target_save_path, self.target_file_name))
            raise "Finish collecting RoI feature"


if __name__ == "__main__":
    from icecream import ic
    from matplotlib import pyplot as plt
    import torch.nn.functional as F
    import math


    import numpy as np
    # roi align
    arg_dict = {
        'roi_align_size': 10,
        'cav_range': [-51.2, -51.2, -3, 51.2, 51.2, 1], # 2x downsample
        'target_roi_num': 10000
    }
    rafe = RoIAlignFeatureExtractor(arg_dict)

    object_bbx_center_single = torch.zeros((1, 100, 7))
    object_bbx_mask_single = torch.zeros((1, 100))

    object_bbx_center_single[0,0] = torch.tensor([0,0,0,2,80,80.0,np.pi/6])
    object_bbx_mask_single[0,0] = 1

    data_dict = {
        'record_len': [1],
        'object_bbx_center_single': object_bbx_center_single,
        'object_bbx_mask_single': object_bbx_mask_single
    }

    # feature
    feature = torch.arange(32*32).float().view(1,1,32,32)
    angle = -30
    # 定义旋转矩阵
    theta = torch.tensor([
        [math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0],
        [math.sin(math.radians(angle)), math.cos(math.radians(angle)), 0]
    ], dtype=torch.float).unsqueeze(0)

    # 生成仿射网格
    grid = F.affine_grid(theta, feature.size())

    # 使用grid_sample进行旋转
    rotated_image = F.grid_sample(feature, grid)
    feature = rotated_image
    roi_num = rafe(data_dict, feature)


    feature_np = feature.squeeze().numpy()
    roi_np = rafe.roi_feature_pool[0].squeeze().numpy()
    plt.imshow(feature_np, vmin=0, vmax=32*32)
    plt.colorbar()
    plt.savefig("opencood/logs_HEAL/vislog/roi_align_rotated/feature.jpg")
    plt.close()
    plt.imshow(roi_np, vmin=0, vmax=32*32)
    plt.colorbar()
    plt.savefig("opencood/logs_HEAL/vislog/roi_align_rotated/roi.jpg")
    plt.close()
