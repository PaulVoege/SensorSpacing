import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce

from opencood.models.fuse_modules.hmvit.base_transformer import (
    HeteroPreNorm,
    HeteroFeedForward,
    HeteroLayerNorm,
    HeteroPreNormResidual,
    CavAttention,
)
from opencood.models.fuse_modules.hmvit.torch_transformation_utils import (
    get_roi_and_cav_mask,
    SpatialTransformation,
)
from opencood.models.fuse_modules.hmvit.hetero_fusion import HeteroAttention
from opencood.models.fuse_modules.hmvit.split_attn import SplitAttn
from natten import NeighborhoodAttention2D
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_


class SDFusionBlock(nn.Module):
    """
    Swap Fusion Block contains window attention and grid attention with
    mask enabled for multi-vehicle cooperation.
    """

    def __init__(self, config):
        super(SDFusionBlock, self).__init__()
        input_dim = config["input_dim"]
        mlp_dim = config["mlp_dim"]
        agent_size = config["agent_size"]
        window_size = config["window_size"]
        drop_out = config["drop_out"]
        dim_head = config["dim_head"]
        num_types = config.get("num_types", 2)
        self.architect_mode = config["architect_mode"]

        if self.architect_mode == "parallel":
            raise NotImplementedError
            self.split_attn = SplitAttn(input_dim, num_windows=2)

        self.spatial_transform = SpatialTransformation(config["spatial_transform"])
        self.downsample_rate = config["spatial_transform"]["downsample_rate"]
        self.discrete_ratio = config["spatial_transform"]["voxel_size"][0]

        self.window_size = window_size

        self.spatial_norm = HeteroLayerNorm(input_dim, num_types)

        self.spatial_attention = NeighborhoodAttention2D(
            input_dim, num_heads=8, kernel_size=7
        )

        self.spatial_ffd = HeteroPreNormResidual(
            input_dim,
            HeteroFeedForward(input_dim, mlp_dim, drop_out, num_types),
            num_types,
        )

        self.depth_norm = HeteroLayerNorm(input_dim, num_types)

        self.depth_attention = CavAttention(
            dim=256, heads=8, dim_head=32, dropout=0.1
        )  # accept [B L H W C]

        self.depth_ffd = HeteroPreNormResidual(
            input_dim,
            HeteroFeedForward(input_dim, mlp_dim, drop_out, num_types),
            num_types,
        )
        self.aggregate_fc = HeteroFeedForward(
            mlp_dim * 3, mlp_dim, drop_out, num_types, out_dim=mlp_dim
        )

    def change_ith_to_first(self, x_pair, mask_pair, mode, i):
        """"""
        L = x_pair.shape[1]
        order = [i] + [j for j in range(L) if j != i]
        x_agent = x_pair[:, order, ...]
        mask_agent = mask_pair[:, :, :, :, order]
        mode_agent = mode[:, order]
        return x_agent, mask_agent, mode_agent

    def warp_features(self, x, pairwise_t_matrix, mask):
        # x: (B, L, C, H, W)
        # pairwise_t_matrix: (B, L, L, 4, 4)
        B, L, C, H, W = x.shape
        x_pair = []
        mask_pair = []
        for i in range(L):
            transformation_matrix = pairwise_t_matrix[:, :, i, :, :]
            x_agent = self.spatial_transform(x, transformation_matrix)
            # (B, H, W, 1, L)
            com_mask = get_roi_and_cav_mask(
                (B, L, H, W, C),
                mask,
                transformation_matrix,
                self.discrete_ratio,
                self.downsample_rate,
            )
            x_pair.append(x_agent)
            mask_pair.append(com_mask)
        # (B, L, L, C, H, W)
        x_pair = torch.stack(x_pair, dim=2)
        # (B, H, W, 1, L, L)
        # mask[...,i,j]: i->j mask
        mask_pair = torch.stack(mask_pair, dim=-1)
        return x_pair, mask_pair

    def spatial_multi_agent_attention(
        self, x, pairwise_t_matrix, mask, mode, record_len, exclude_self
    ):
        # x: b l c h w
        # mask: b, l
        # window attention -> grid attention
        x_normed = self.spatial_norm(x.permute(0, 1, 3, 4, 2), mode)  # b l h w c
        B, L, H, W, C = x_normed.shape
        x_normed = x_normed.flatten(0, 1)  # b*l h w c
        x_normed = self.spatial_attention(x_normed)
        x_normed = x_normed.view(B, L, H, W, C) + x.permute(0, 1, 3, 4, 2)

        x = self.spatial_ffd(x_normed, mode).permute(0, 1, 4, 2, 3)

        return x

    def depth_multi_agent_attention(
        self, x, pairwise_t_matrix, mask, mode, record_len, exclude_self
    ):
        # grid attention
        # x: b l c h w
        # mask: b,l
        # window attention -> grid attention
        x_normed = self.depth_norm(x.permute(0, 1, 3, 4, 2), mode).permute(
            0, 1, 4, 2, 3
        )
        x_pair, mask_pair = self.warp_features(x_normed, pairwise_t_matrix, mask)

        max_cav = record_len.max()
        B, L = pairwise_t_matrix.shape[:2]
        x_updated = []
        for i in range(max_cav):  # each agent as ego
            # ic| x_agent.shape: torch.Size([1, 2, 256, 128, 128]) # [B, L, C, H, W]
            # ic| mask_agent.shape: torch.Size([1, 128, 128, 1, 2]) # [B, H, W, 1, L]
            # ic| mode_agent.shape: torch.Size([1, 2]) # [B, L]
            x_agent, mask_agent, mode_agent = self.change_ith_to_first(
                x_pair[:, :max_cav, i, ...],
                mask_pair[:, :, :, :, :max_cav, i],
                mode[:, :max_cav],
                i,
            )
            x_agent = x_agent  # [B, L, C, H, W]
            mask_swap = mask_agent  # [B, H, W, 1, L]
            x_agent = self.depth_attention(x_agent, mask=mask_swap)  # [B, 1, H, W, C]
            x_agent = x_agent.permute(0, 1, 4, 2, 3)  # [B, 1, C, H, W]

            x_updated.append(x_agent)
        x_updated = torch.cat(x_updated, dim=1)
        # (B, L, C, H, W)
        x = x = F.pad(x_updated, (0, 0, 0, 0, 0, 0, 0, L - max_cav, 0, 0)) + x

        x = self.depth_ffd(x.permute(0, 1, 3, 4, 2), mode).permute(0, 1, 4, 2, 3)
        return x

    def spatial_depth_multi_agent_attention(
        self, x, pairwise_t_matrix, mask, mode, record_len, exclude_self
    ):
        # x: b l c h w
        ############### Spatial Attention ###############
        x_spatial_normed = self.spatial_norm(
            x.permute(0, 1, 3, 4, 2), mode
        )  # b l h w c
        B, L, H, W, C = x_spatial_normed.shape
        x_spatial_normed = x_spatial_normed.flatten(0, 1)  # b*l h w c
        x_spatial_normed = self.spatial_attention(x_spatial_normed)
        x = x_spatial_normed.view(B, L, H, W, C) + x.permute(0, 1, 3, 4, 2)
        x = x.permute(0, 1, 4, 2, 3)

        # x: b l c h w
        # mask: b,l
        # ############### Depth Attention ###############
        x_normed = self.depth_norm(x.permute(0, 1, 3, 4, 2), mode).permute(
            0, 1, 4, 2, 3
        )
        x_pair, mask_pair = self.warp_features(x_normed, pairwise_t_matrix, mask)

        max_cav = record_len.max()
        B, L = pairwise_t_matrix.shape[:2]
        x_updated = []
        for i in range(max_cav):  # each agent as ego
            # ic| x_agent.shape: torch.Size([1, 2, 256, 128, 128]) # [B, L, C, H, W]
            # ic| mask_agent.shape: torch.Size([1, 128, 128, 1, 2]) # [B, H, W, 1, L]
            # ic| mode_agent.shape: torch.Size([1, 2]) # [B, L]
            x_agent, mask_agent, mode_agent = self.change_ith_to_first(
                x_pair[:, :max_cav, i, ...],
                mask_pair[:, :, :, :, :max_cav, i],
                mode[:, :max_cav],
                i,
            )
            x_agent = x_agent  # [B, L, C, H, W]
            mask_swap = mask_agent  # [B, H, W, 1, L]
            x_agent = self.depth_attention(x_agent, mask=mask_swap)  # [B, 1, H, W, C]
            x_agent = x_agent.permute(0, 1, 4, 2, 3)  # [B, 1, C, H, W]

            x_updated.append(x_agent)
        x_updated = torch.cat(x_updated, dim=1)
        # (B, L, C, H, W)
        x = x = F.pad(x_updated, (0, 0, 0, 0, 0, 0, 0, L - max_cav, 0, 0)) + x

        x = self.depth_ffd(x.permute(0, 1, 3, 4, 2), mode).permute(0, 1, 4, 2, 3)

        return x

    def forward(self, x, pairwise_t_matrix, mode, record_len, mask):
        # x: (B, L, C, H, W)
        # pairwise_t_matrix: (B, L, L, 4, 4)
        # mask: (B, H, W, 1, L, L)

        if self.architect_mode == "sequential":
            x = self.spatial_multi_agent_attention(
                x, pairwise_t_matrix, mask, mode, record_len, exclude_self=False
            )
            x = self.depth_multi_agent_attention(
                x, pairwise_t_matrix, mask, mode, record_len, exclude_self=False
            )
        elif self.architect_mode == "sd_in_one_block":
            x = self.spatial_depth_multi_agent_attention(
                x, pairwise_t_matrix, mask, mode, record_len, exclude_self=False
            )

        else:
            raise ValueError(f"{self.architect_mode} not implemented")

        return x


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = {
        "input_dim": 512,
        "mlp_dim": 512,
        "agent_size": 4,
        "window_size": 8,
        "dim_head": 4,
        "drop_out": 0.1,
        "depth": 2,
        "mask": True,
    }
    block = HeteroFusionBlock(args)
    block.cuda()
    test_data = torch.rand(1, 4, 512, 32, 32)
    test_data = test_data.cuda()
    mask = torch.ones(1, 32, 32, 1, 4)
    mask = mask.cuda()

    output = block(test_data, mask)
    print(output)
