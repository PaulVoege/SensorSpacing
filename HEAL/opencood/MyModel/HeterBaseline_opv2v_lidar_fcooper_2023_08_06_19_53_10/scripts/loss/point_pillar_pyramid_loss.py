import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.loss.point_pillar_depth_loss import PointPillarDepthLoss
from opencood.loss.point_pillar_loss import sigmoid_focal_loss

class PointPillarPyramidLoss(PointPillarDepthLoss):
    def __init__(self, args):
        super().__init__(args)
        self.pyramid = args['pyramid']

        # relative downsampled GT cls map from fused labels.
        self.relative_downsample = self.pyramid['relative_downsample']
        self.pyramid_weight = self.pyramid['weight']
        self.num_levels = len(self.relative_downsample)
    


    def forward(self, output_dict, target_dict, suffix=""):
        if suffix == "":
            return super().forward(output_dict, target_dict)

        assert suffix == "_single"

        if 'record_len' in output_dict:
            batch_size = int(output_dict['record_len'].sum())
        elif 'batch_size' in output_dict:
            batch_size = output_dict['batch_size']
        else:
            batch_size = target_dict['pos_equal_one'].shape[0]

        positives = target_dict['pos_equal_one']
        negatives = target_dict['neg_equal_one']

        cls_preds_single_list = output_dict['cls_preds_single_list']

        total_cls_loss = 0

        for i, cls_preds_single in enumerate(cls_preds_single_list):
            positives_level = F.max_pool2d(positives.permute(0,3,1,2), kernel_size=self.relative_downsample[i]).permute(0,2,3,1)
            negatives_level = F.max_pool2d(negatives.permute(0,3,1,2), kernel_size=self.relative_downsample[i]).permute(0,2,3,1)

            cls_labls = positives_level.view(batch_size, -1, 1)
            positives_level = cls_labls > 0
            negatives_level = negatives_level.view(batch_size, -1, 1) > 0

            pos_normalizer = positives_level.sum(1, keepdim=True).float()


            # cls loss
            cls_preds = cls_preds_single.permute(0, 2, 3, 1).contiguous() \
                        .view(batch_size, -1,  1)
            cls_weights = positives_level * self.pos_cls_weight + negatives_level * 1.0
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_loss = sigmoid_focal_loss(cls_preds, cls_labls, weights=cls_weights, **self.cls)
            cls_loss = cls_loss.sum() * self.cls['weight'] / batch_size
            cls_loss *= self.pyramid_weight[i]

            total_cls_loss += cls_loss

        self.loss_dict = {"total_loss": total_cls_loss,
                          "cls_loss": total_cls_loss}
        
        return total_cls_loss

