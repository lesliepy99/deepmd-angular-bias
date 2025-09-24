# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Loss: Force + Relative-coordinate + Cos(theta)
"""

from typing import Optional

import torch
import torch.nn.functional as F

from deepmd.pt.loss.loss import TaskLoss
from deepmd.pt.utils import env
from deepmd.pt.utils.env import GLOBAL_PT_FLOAT_PRECISION
from deepmd.utils.data import DataRequirementItem


class PretrainLoss(TaskLoss):
    r"""
    联合损失:
        (1) 原子力          force
        (2) 邻居相对坐标    rel_coord
        (3) 连续邻居余弦    cos_theta       ← 新增

    关键参数
    ----------
    start_pref_f / limit_pref_f   ---- 力损失前因子
    start_pref_rc / limit_pref_rc ---- rel_coord 损失前因子
    start_pref_ct / limit_pref_ct ---- cos_theta 损失前因子   ← 新增
    relative_f                    ---- 若给定则使用相对力误差
    use_l1                        ---- True: L1   False: L2
    """

    # -------------------------------------------------------------- #
    # 初始化                                                          #
    # -------------------------------------------------------------- #
    def __init__(
        self,
        starter_learning_rate: float = 1.0,
        # force
        start_pref_f: float = 0.0,
        limit_pref_f: float = 0.0,
        # rel-coord
        start_pref_rc: float = 0.0,
        limit_pref_rc: float = 0.0,
        # cosθ
        start_pref_ct: float = 0.0,              # ← 新增
        limit_pref_ct: float = 0.0,              # ← 新增
        # misc
        relative_f: Optional[float] = None,
        use_l1: bool = False,
        inference: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.starter_learning_rate = starter_learning_rate

        # 哪些分支被启用
        self.has_f  = (start_pref_f  != 0.0 and limit_pref_f  != 0.0) or inference
        self.has_rc = (start_pref_rc != 0.0 and limit_pref_rc != 0.0) or inference
        self.has_ct = (start_pref_ct != 0.0 and limit_pref_ct != 0.0) or inference

        # 记录前因子
        self.start_pref_f,  self.limit_pref_f  = start_pref_f,  limit_pref_f
        self.start_pref_rc, self.limit_pref_rc = start_pref_rc, limit_pref_rc
        self.start_pref_ct, self.limit_pref_ct = start_pref_ct, limit_pref_ct

        self.relative_f = relative_f
        self.use_l1     = use_l1
        self.inference  = inference

    # -------------------------------------------------------------- #
    # forward                                                        #
    # -------------------------------------------------------------- #
    def forward(
        self,
        input_dict: dict,
        model,
        label: dict,
        natoms: int,
        learning_rate: float,
        mae: bool = False,
    ):
        model_pred = model(**input_dict)

        coef     = learning_rate / self.starter_learning_rate
        pref_f   = self.limit_pref_f  + (self.start_pref_f  - self.limit_pref_f)  * coef
        pref_rc  = self.limit_pref_rc + (self.start_pref_rc - self.limit_pref_rc) * coef
        pref_ct  = self.limit_pref_ct + (self.start_pref_ct - self.limit_pref_ct) * coef

        loss = torch.zeros((), dtype=GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)
        more = {}

        # ==========================================================
        # (1) Force loss
        # ==========================================================
        if self.has_f and "force" in model_pred and "force" in label:
            find_force = label.get("find_force", 0.0)
            pref_f = pref_f * find_force

            force_p = model_pred["force"]            # [nf,nloc,3]
            force_l = label["force"]
            diff_f  = force_l - force_p

            # 相对误差
            if self.relative_f is not None:
                norm   = force_l.norm(dim=-1, keepdim=True) + self.relative_f
                diff_f = diff_f / norm

            if not self.use_l1:
                l2_f = torch.mean(diff_f.pow(2))
                if not self.inference:
                    more["l2_force_loss"] = self.display_if_exist(l2_f.detach(),
                                                                   find_force)
                loss += pref_f * l2_f
                more["rmse_f"] = self.display_if_exist(l2_f.sqrt().detach(),
                                                       find_force)
            else:
                l1_f  = F.l1_loss(force_p, force_l, reduction="none")
                mae_f = l1_f.mean()
                more["mae_f"] = self.display_if_exist(mae_f.detach(), find_force)
                loss += pref_f * l1_f.sum()

            if mae:
                more["mae_f_all"] = self.display_if_exist(
                    torch.mean(torch.abs(diff_f)).detach(), find_force
                )

        # ==========================================================
        # (2) rel_coord loss
        # ==========================================================
        if self.has_rc and "rel_coord" in model_pred and "gt_rel_coord" in model_pred:
            find_rc = 1.0
            pref_rc = pref_rc * find_rc

            rc_p = model_pred["rel_coord"]            # [nf,nloc,nsel,3]
            rc_l = model_pred["gt_rel_coord"]

            mask = model_pred.get("neighbor_mask", None)
            if mask is not None:
                mask = mask.to(rc_p.dtype).unsqueeze(-1)
                diff_rc  = (rc_l - rc_p) * mask
                norm_fac = mask.mean() + 1e-8
            else:
                diff_rc  = rc_l - rc_p
                norm_fac = 1.0

            if not self.use_l1:
                l2_rc = torch.mean(diff_rc.pow(2)) / norm_fac
                if not self.inference:
                    more["l2_relcoord_loss"] = self.display_if_exist(
                        l2_rc.detach(), find_rc
                    )
                loss += pref_rc * l2_rc
                more["rmse_rc"] = self.display_if_exist(l2_rc.sqrt().detach(), find_rc)
            else:
                l1_rc = torch.mean(torch.abs(diff_rc)) / norm_fac
                more["mae_rc"] = self.display_if_exist(l1_rc.detach(), find_rc)
                loss += pref_rc * l1_rc

        # ==========================================================
        # (3) cos_theta loss          ← 新增分支
        # ==========================================================
        K = 6
        if self.has_ct and "cos_theta" in model_pred and "gt_cos_theta" in model_pred:
            find_ct = 1.0
            pref_ct = pref_ct * find_ct

            ct_p = model_pred["cos_theta"]            # [nf,nloc,nsel-1]
            ct_l = model_pred["gt_cos_theta"]
            # print(ct_p.shape)
            # print(ct_l.shape)
            ct_p = ct_p[:,:,:K]
            ct_l = ct_l[:,:,:K]
            # print(ct_p.shape)
            # print(ct_l.shape)
            # exit()
           
            mask = model_pred.get("triplet_mask", None)
           
            mask = mask[:,:,:K]
            if mask is not None:
                mask = mask.to(ct_p.dtype)
                diff_ct  = (ct_l - ct_p) * mask
                norm_fac = mask.mean() + 1e-8
            else:
                diff_ct  = ct_l - ct_p
                norm_fac = 1.0

            if not self.use_l1:
                l2_ct = torch.mean(diff_ct.pow(2)) / norm_fac
                if not self.inference:
                    more["l2_costheta_loss"] = self.display_if_exist(
                        l2_ct.detach(), find_ct
                    )
                loss += pref_ct * l2_ct
                more["rmse_ct"] = self.display_if_exist(l2_ct.sqrt().detach(), find_ct)
            else:
                l1_ct = torch.mean(torch.abs(diff_ct)) / norm_fac
                more["mae_ct"] = self.display_if_exist(l1_ct.detach(), find_ct)
                loss += pref_ct * l1_ct

        # ----------------------------------------------------------
        # 总体 rmse
        # ----------------------------------------------------------
        if not self.inference:
            more["rmse"] = torch.sqrt(loss.detach())

        return model_pred, loss.to(GLOBAL_PT_FLOAT_PRECISION), more

    # -------------------------------------------------------------- #
    # 数据集标签需求                                                 #
    # -------------------------------------------------------------- #
    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        req: list[DataRequirementItem] = []
        if self.has_f:
            req.append(
                DataRequirementItem("force", ndof=3, atomic=True,
                                    must=False, high_prec=False)
            )
        if self.has_rc:
            req.append(
                DataRequirementItem("rel_coord", ndof=3, atomic=True,
                                    must=False, high_prec=False)
            )
            req.append(
                DataRequirementItem("neighbor_mask", ndof=1, atomic=True,
                                    must=False, high_prec=False, default=1)
            )
        # cos_theta 的真值已在模型 forward 中动态生成，
        # 如果你想从数据集直接读，也可在此添加 DataRequirementItem。
        return req