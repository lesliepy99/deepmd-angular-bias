
from __future__ import annotations
from typing import Optional, List, Tuple

import torch
import torch.nn as nn

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
    fitting_check_output,
)
from deepmd.pt.model.network.network import NonLinearHead
from deepmd.pt.model.task.fitting import Fitting
from deepmd.pt.model.task.base_fitting import BaseFitting
from deepmd.utils.version import check_version_compatibility

__all__ = ["NeighborCoordFitting"]


@BaseFitting.register("neighbor_coord")
@fitting_check_output
class NeighborCoordFitting(Fitting):
    """
    输入 : pair-wise 描述符 g_ij, 形状 (nf , nloc , nsel , dim_pair)
    预测 : 1) 相对坐标 Δr_ij  (nsel , 3)
           2) 连续邻居 (k,k+1) 的 cosθ      (nsel-1)
    """

    # ------------------------------------------------------------ #
    # 初始化                                                        #
    # ------------------------------------------------------------ #
    def __init__(
        self,
        dim_pair: int,
        nsel: int,
        hidden: Optional[int | List[int]] = None,
        hidden_angle: Optional[int | List[int]] = None,   # <-- 新增
        activation_function: str = "gelu",
        **kwargs,
    ) -> None:
        super().__init__()
        self.dim_pair = int(dim_pair)
        self.nsel = int(nsel)
        self.K = 6

        # ---------------- 相对坐标 MLP ---------------- #
        if hidden is None:
            hidden = [128, 128]
        if isinstance(hidden, int):
            hidden = [hidden]
        self.hidden = list(hidden)
        self.activation_function = activation_function

        layers: list[nn.Module] = []
        in_dim = self.dim_pair
        for width in self.hidden:
            layers.append(
                NonLinearHead(
                    in_dim,
                    width,
                    activation_fn=self.activation_function,
                )
            )
            in_dim = width
        layers.append(nn.Linear(in_dim, 3))
        self.mlp = nn.Sequential(*layers).double()

        # ---------------- 三邻居角度 MLP --------------- #
        if hidden_angle is None:
            hidden_angle = [128, 64]
        if isinstance(hidden_angle, int):
            hidden_angle = [hidden_angle]
        self.hidden_angle = list(hidden_angle)

        layers_ang: list[nn.Module] = []
        in_dim = 2 * self.dim_pair                    # concat(k,k+1)
        for width in self.hidden_angle:
            layers_ang.append(
                NonLinearHead(
                    in_dim,
                    width,
                    activation_fn=self.activation_function,
                )
            )
            in_dim = width
        # 输出 1 个标量, tanh → (-1,1)
        layers_ang.append(nn.Linear(in_dim, 1))
        layers_ang.append(nn.Tanh())
        self.mlp_angle = nn.Sequential(*layers_ang).double()

    # ------------------------------------------------------------ #
    # output_def                                                   #
    # ------------------------------------------------------------ #
    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(         # Δr_ij 预测
                    "rel_coord",
                    [self.nsel, 3],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
                OutputVariableDef(         # Δr_ij 真值
                    "gt_rel_coord",
                    [self.nsel, 3],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
                OutputVariableDef(         # 邻居存在掩码
                    "neighbor_mask",
                    [self.nsel],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
                # ------------ 新增三邻居角相关张量 ------------
                OutputVariableDef(
                    "cos_theta",           # 预测 cosθ
                    [self.K],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
                OutputVariableDef(
                    "gt_cos_theta",        # 真值 cosθ (可留空由 dataloader 提供)
                    [self.K],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
                OutputVariableDef(
                    "triplet_mask",        # (k,k+1) 均存在
                    [self.K],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
            ]
        )

    # ------------------------------------------------------------ #
    # 与元素类型无关                                                #
    # ------------------------------------------------------------ #
    def get_type_map(self) -> list[str]:
        return []

    def change_type_map(self, type_map: list[str], model_with_new_type_stat=None):
        pass

    # ------------------------------------------------------------ #
    # 序列化 / 反序列化                                             #
    # ------------------------------------------------------------ #
    def serialize(self) -> dict:
        return {
            "@class": "NeighborCoordFitting",
            "@version": 3,
            "dim_pair": self.dim_pair,
            "nsel": self.nsel,
            "hidden": self.hidden,
            "hidden_angle": self.hidden_angle,
            "activation": self.activation_function,
        }

    @classmethod
    def deserialize(cls, data) -> "NeighborCoordFitting":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 3), 3, 3)
        data.pop("@class", None)
        return cls(**data)

    # ------------------------------------------------------------ #
    # forward                                                      #
    # ------------------------------------------------------------ #
    def forward(
        self,
        gg: torch.Tensor,             # (nf , nloc , nsel , dim_pair)
        gt_rel_coord: torch.Tensor,   # (nf , nloc , nsel , 3)
        nlist: torch.Tensor,          # (nf , nloc , nsel)
        gt_cos_theta: Optional[torch.Tensor] = None,   # 如有提供
        **kwargs,
    ) -> dict[str, torch.Tensor]:

        dtype_net = next(self.parameters()).dtype
        gg = gg.to(dtype_net)

        nf, nloc, nsel, dim_pair = gg.shape
        assert nsel == self.nsel and dim_pair == self.dim_pair, \
            "输入维度与模型配置不一致"

        # ---------- Δr_ij head ----------
        x = gg.reshape(-1, dim_pair)                 # (nf*nloc*nsel , dim_pair)
        rel_out = self.mlp(x).reshape(nf, nloc, nsel, 3)

        neighbor_mask = (nlist >= 0).to(dtype_net)   # (nf , nloc , nsel)

        # ---------- cosθ head -----------
        if nsel < 2:
            raise ValueError("nsel must ≥2 to build neighbor pairs")


        K = self.K
        # 1) 构造 (k,k+1) pair
        # feat_left  = gg[:, :, :-1, :]
        # feat_right = gg[:, :, 1:,  :]

        # 第 0 个元素复制成与其余元素同长度
        feat_left  = gg[:, :, 0:1, :].expand(-1, -1, nsel-1, -1)   # (nf,n_loc,n_sel-1,d)

        # 其余元素 1…n_sel-1
        feat_right = gg[:, :, 1:, :]                                 # (nf,n_loc,n_sel-1,d)

        feat_pair  = torch.cat([feat_left, feat_right], dim=-1)      # (nf,nloc,nsel-1,2*dim_pair)

        # 2) 裁剪到固定 K
        K = min(K, feat_pair.shape[2])
        feat_pair   = feat_pair[:, :, :K, :]                         # (nf,nloc,K,2*dim_pair)
        x_pair      = feat_pair.reshape(-1, 2*dim_pair)

        # 3) 预测 cosθ
        cos_out = self.mlp_angle(x_pair).view(nf, nloc, K)
        cos_out = cos_out.view(nf, nloc, K)

        # 4) 同步裁剪 triplet_mask / gt_cos_theta
        triplet_mask = neighbor_mask[:, :, :-1] * neighbor_mask[:, :, 1:]
        triplet_mask = triplet_mask[:, :, :K].to(dtype_net)

        # # 1. 构造 (k,k+1) 特征   → (nf,nloc,nsel-1,2*dim_pair)
        # feat_left  = gg[:, :, :-1, :]           # k
        # feat_right = gg[:, :, 1:,  :]           # k+1
        # feat_pair  = torch.cat([feat_left, feat_right], dim=-1)

        # # 2. 送入 MLP
        # x_pair = feat_pair.reshape(-1, 2 * dim_pair)
        # cos_out = self.mlp_angle(x_pair)        # tanh → (-1,1)
        # cos_out = cos_out.view(nf, nloc, nsel - 1)

        # # 3. 掩码：两邻居都存在
        # triplet_mask = neighbor_mask[:, :, :-1] * neighbor_mask[:, :, 1:]
        # triplet_mask = triplet_mask.to(dtype_net)     # 便于后续 loss 加权

       

        # 若 dataloader 未提供 gt_cos_theta，可在此处通过 Δr 计算
        if gt_cos_theta is None:
            # 用真实 Δr 计算
            v1 = gt_rel_coord[:, :, :-1, :]           # (k)
            v2 = gt_rel_coord[:, :, 1:,  :]           # (k+1)
            dot = (v1 * v2).sum(-1)
            n1 = v1.norm(dim=-1)
            n2 = v2.norm(dim=-1)
            gt_cos_theta = dot / (n1 * n2 + 1e-12)
            gt_cos_theta = gt_cos_theta.where(triplet_mask.bool(), torch.zeros_like(gt_cos_theta))
        gt_cos_theta = gt_cos_theta[:, :, :K]

        # print(cos_out.shape)
        # print(gt_cos_theta.shape)
        # exit()
        # print(cos_out)
        return {
            "rel_coord":      rel_out,
            "gt_rel_coord":   gt_rel_coord,
            "neighbor_mask":  neighbor_mask,
            "cos_theta":      cos_out,
            "gt_cos_theta":   gt_cos_theta.to(dtype_net),
            "triplet_mask":   triplet_mask,
        }