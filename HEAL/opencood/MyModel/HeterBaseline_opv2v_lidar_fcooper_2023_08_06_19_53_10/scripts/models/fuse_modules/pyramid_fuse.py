"""
Pyramid Fuse Module. 

Similar to the ResNetBEVBackbone
"""

import numpy as np
import torch
import torch.nn as nn

from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.resblock import ResNetModified, Bottleneck, BasicBlock
from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.visualization.debug_plot import plot_feature

def normalize_score(score, record_len):
    """
    Parameters
    ----------
    score : torch.Tensor
        shape: (sum(n_cav), 2, H, W)
        
    record_len : list
        shape: (B)
    """
    score = torch.sum(score, dim=1, keepdim=True)
    split_score = regroup(score, record_len)
    split_score = [torch.softmax(s, dim=0) for s in split_score]
    score = torch.cat(split_score, dim=0)
    return score

def weighted_fuse(x, score, record_len, affine_matrix):
    """
    Parameters
    ----------
    x : torch.Tensor
        input data, (sum(n_cav), C, H, W)
    
    weight : torch.Tensor
        weight for weighted sum, (sum(n_cav), 2, H, W)
        
    record_len : list
        shape: (B)
        
    affine_matrix : torch.Tensor
        normalized affine matrix from 'normalize_pairwise_tfm'
        shape: (B, L, L, 2, 3) 
    """

    _, C, H, W = x.shape
    B, L = affine_matrix.shape[:2]
    split_x = regroup(x, record_len)
    score = torch.sum(score, dim=1, keepdim=True)
    split_score = regroup(score, record_len)
    batch_node_features = split_x
    out = []
    # iterate each batch
    for b in range(B):
        N = record_len[b]
        score = split_score[b]
        t_matrix = affine_matrix[b][:N, :N, :, :]
        i = 0 # ego
        feature_in_ego = warp_affine_simple(batch_node_features[b],
                                        t_matrix[i, :, :, :],
                                        (H, W))
        scores_in_ego = warp_affine_simple(split_score[b],
                                           t_matrix[i, :, :, :],
                                           (H, W))
        scores_in_ego = torch.softmax(scores_in_ego, dim=0)

        out.append(torch.sum(feature_in_ego * scores_in_ego, dim=0))
    out = torch.stack(out)
    
    return out

class PyramidFusion(ResNetBEVBackbone):
    def __init__(self, model_cfg, input_channels=64):
        """
        Do not downsample in the first layer.
        """
        super().__init__(model_cfg, input_channels)
        if model_cfg["resnext"]:
            Bottleneck.expansion = 1
            self.resnet = ResNetModified(Bottleneck, 
                                        self.model_cfg['layer_nums'],
                                        self.model_cfg['layer_strides'],
                                        self.model_cfg['num_filters'],
                                        inplanes = model_cfg.get('inplanes', 64),
                                        groups=32,
                                        width_per_group=4)
        
        # add single supervision head
        for i in range(self.num_levels):
            setattr(
                self,
                f"single_head_{i}",
                nn.Conv2d(self.model_cfg["num_filters"][i], model_cfg['anchor_number'], kernel_size=1),
            )

    def forward_single(self, spatial_features):
        """
        This is used for single agent pass.
        """

        return self.forward({"spatial_features": spatial_features})['spatial_features_2d']
    
    def forward_collab(self, spatial_features, record_len, affine_matrix):
        """
        
        """
        feature_list = self.get_multiscale_feature(spatial_features)
        fused_feature_list = []
        cls_map_list = []
        for i in range(self.num_levels):
            cls_map = eval(f"self.single_head_{i}")(feature_list[i])
            cls_map_list.append(cls_map)
            score = torch.sigmoid(cls_map)
            fused_feature_list.append(weighted_fuse(feature_list[i], score, record_len, affine_matrix))
        fused_feature = self.decode_multiscale_feature(fused_feature_list)

        
        return fused_feature, cls_map_list 