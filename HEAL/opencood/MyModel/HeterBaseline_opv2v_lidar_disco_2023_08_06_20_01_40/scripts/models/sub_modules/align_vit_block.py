import torch
import torch.nn.functional as F
from mmcv.ops import DeformConv2dPack as dconv2d
from natten import (NeighborhoodAttention2D, NeighborhoodAttention3D,
                    natten2dav, natten2dqkrpb, natten3dav, natten3dqkrpb)
from torch import einsum, nn
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_

from opencood.models.fuse_modules.fusion_in_one import regroup, warp_feature
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.visualization.debug_plot import plot_feature

class AlignViTBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args["dim"]
        dim_head = args["dim_head"]  # per head dim
        kernel_size = args["kernel_size"]

        self.na = NeighborhoodAttention2D(dim=dim, kernel_size=kernel_size, dilation=1, num_heads=dim//dim_head)

        self.norm1 = nn.LayerNorm(dim)  # channel last
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)

        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.linear3 = nn.Linear(dim, dim)
        self.linear4 = nn.Linear(dim, dim)

        self.act = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x_residual = x
        x = self.norm1(x)
        x = self.na(x)
        x = self.norm2(x)
        x = self.linear1(x) # N,H*W,C
        x = x + x_residual

        x_residual = x
        x = self.norm3(x)
        x = self.linear2(x)
        x = self.act(x)
        x = self.norm4(x)
        x = self.linear3(x)
        x = x + x_residual

        x = x.permute(0,3,1,2)

        return x
    

class AdaptiveNeighborhoodAttention3D(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size,
        dilation=1,
        bias=True,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        dilation = dilation or 1
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert (
            dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."

        self.dilation_d = 1
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.register_parameter("rpb", None)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask):
        """
        Args:
            x: torch.tensor
                shape [B, D, H, W, C], depth is our cav number
            mask: torch.tensor
                shape [B, D, H, W, 1], depth is our cav number
        Returns:
            x: torch.tensor
                shape [B, D, H, W, C]
        """
        B, D, H, W, C = x.shape
        input_D = D

        if D == 1:
            return x

        if D % 2 == 0:
            pad_x = torch.zeros(B, 1, H, W, C).to(x)
            pad_mask = torch.zeros(B, 1, H, W, 1).to(mask)
            x = torch.cat((x, pad_x), dim=1)
            mask = torch.cat((mask, pad_mask), dim=1)
            B, D, H, W, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, D, H, W, 3, self.num_heads, self.head_dim)
            .permute(4, 0, 5, 1, 2, 3, 6)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = natten3dqkrpb(
            q,
            k,
            self.rpb,
            D,
            self.kernel_size,
            self.dilation_d,
            self.dilation,
        )

        # attn.shape = [B, n_head, D, H, W, ks * ks * ks_d], ks is kernel_size
        mask = mask.unsqueeze(1) # broadcast for the head
        attn = attn * mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = natten3dav(
            attn,
            v,
            D,
            self.kernel_size,
            self.dilation_d,
            self.dilation,
        )
        x = x.permute(0, 2, 3, 4, 1, 5).reshape(B, D, H, W, C)

        x = x[:, :input_D, :, :, :]

        return self.proj_drop(self.proj(x))


class AlignViTFusionBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args["dim"]
        dim_head = args["dim_head"]  # per head dim
        kernel_size = args["kernel_size"]
        self.loop_cav = args.get("loop_cav", False)

        self.na = AdaptiveNeighborhoodAttention3D(dim=dim, 
                                                 kernel_size=kernel_size, 
                                                 dilation=1, 
                                                 num_heads=dim//dim_head)

        self.norm1 = nn.LayerNorm(dim)  # channel last
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)

        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.linear3 = nn.Linear(dim, dim)
        self.linear4 = nn.Linear(dim, dim)

        self.act = nn.ReLU()
    
    def sample_ith_to_first(self, x, sample_affine_matrix):
        """
        Let each cav to be ego
        ---------
        Args:
            x : torch.tensor
                shape [l, C, H, W], l is number of cavs in this sample

            sample_affine_matrix : torch.tensor
                shape [L, L, 2, 3]
        Output:
            output_x : torch.tensor
                shape [l, l, C, H, W]
            output_mask : torch.tensor 
                shape [l, l, C, H, W]
        """
        l, C, H, W = x.shape
        mask = torch.ones(l, 1, H, W).to(x)
        t_matrix = sample_affine_matrix[:l, :l, :, :]

        output_x = []
        output_mask = []
        for i in range(l): # i indicate ego
            feature_in_ego = warp_affine_simple(x, t_matrix[i, :, :, :], (H, W)) # [l, C, H, W]
            mask_in_ego = warp_affine_simple(mask, t_matrix[i, :, :, :], (H, W)) # [l, 1, H, W]

            order = [i] + [j for j in range(l) if j != i]
            # move ego to the first
            feature_in_ego = feature_in_ego[order]
            mask_in_ego = mask_in_ego[order]

            output_x.append(feature_in_ego)
            output_mask.append(mask_in_ego)

        output_x = torch.stack(output_x)
        output_mask = torch.stack(output_mask)

        return output_x, output_mask 


    def forward(self, x, record_len, affine_matrix):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)
            
        record_len : list
            shape: (B)
            
        affine_matrix : torch.Tensor
            normalized affine matrix from 'normalize_pairwise_tfm'
            shape: (B, L, L, 2, 3) 
        """
        x = x.permute(0,2,3,1)
        x_residual = x
        x = self.norm1(x)

        B, L = affine_matrix.shape[:2]
        split_x = regroup(x.permute(0,3,1,2), record_len)
        updated_x = []

        for b in range(B):
            x = split_x[b]
            sample_affine_matrix = affine_matrix[b]

            # [l, l, C, H, W] and [l, l, 1, H, W], we treat first l as Batch in self.na
            output_x, output_mask = self.sample_ith_to_first(x, sample_affine_matrix)

            output_x = output_x.permute(0,1,3,4,2) # [l, l, H, W, C]
            output_mask = output_mask.permute(0,1,3,4,2) # [l, l, H, W, 1]
            output_mask = output_mask > 0

            if self.loop_cav:
                for ll in range(len(x)):
                    na_output = self.na(output_x[ll:ll+1], output_mask[ll:ll+1]) # [1, l, H, W, C]
                    updated_x.append(na_output[:,0,:,:,:]) # 0 in the second dimension is always the ego, since it is moved to the first
            else:
                na_output = self.na(output_x, output_mask) # [l, l, H, W, C]
                updated_x.append(na_output[:,0,:,:,:]) # 0 in the second dimension is always the ego, since it is moved to the first
            

        updated_x = torch.cat(updated_x, dim=0) # [N, H, W, C]

        x = updated_x
        x = self.norm2(x)
        x = self.linear1(x) 
        x = x + x_residual

        x_residual = x
        x = self.norm3(x)
        x = self.linear2(x)
        x = self.act(x)
        x = self.norm4(x)
        x = self.linear3(x)
        x = x + x_residual
        x = x.permute(0,3,1,2)

        return x
    
    def forward_short(self, x, record_len, affine_matrix):
        x = x.permute(0,2,3,1)
        x_residual = x
        x = self.norm1(x)

        B, L = affine_matrix.shape[:2]
        split_x = regroup(x.permute(0,3,1,2), record_len)
        updated_x = []

        for b in range(B):
            x = split_x[b]
            sample_affine_matrix = affine_matrix[b]

            # [l, l, C, H, W] and [l, l, 1, H, W], we treat first l as Batch in self.na
            output_x, output_mask = self.sample_ith_to_first(x, sample_affine_matrix)

            output_x = output_x.permute(0,1,3,4,2) # [l, l, H, W, C]
            output_mask = output_mask.permute(0,1,3,4,2) # [l, l, H, W, 1]
            output_mask = output_mask > 0

            if self.loop_cav:
                for ll in range(len(x)):
                    na_output = self.na(output_x[ll:ll+1], output_mask[ll:ll+1]) # [1, l, H, W, C]
                    updated_x.append(na_output[:,0,:,:,:]) # 0 in the second dimension is always the ego, since it is moved to the first
            else:
                na_output = self.na(output_x, output_mask) # [l, l, H, W, C]
                updated_x.append(na_output[:,0,:,:,:]) # 0 in the second dimension is always the ego, since it is moved to the first
            

        updated_x = torch.cat(updated_x, dim=0) # [N, H, W, C]

        x = updated_x
        x = x + x_residual

        x_residual = x
        x = self.norm2(x)
        x = self.linear1(x)
        x = x + x_residual
        x = x.permute(0,3,1,2)


if __name__ == "__main__":
    args = {
        "dim": 256,
        "dim_head": 32,
        "kernel_size": 5,
        "loop_cav": False
    }

    # block = AlignViTBlock(args).cuda()
    # dummy_input = torch.randn(4, 256, 256, 256).cuda()
    # output = block(dummy_input)

    dummy_input = torch.randn(5, 256, 256, 256).cuda()
    record_len = torch.tensor([3,2])
    affine_matrix = torch.randn(2, 5, 5, 2, 3)
    block = AlignViTFusionBlock(args).cuda()
    output = block(dummy_input, record_len, affine_matrix)
