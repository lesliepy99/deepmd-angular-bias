# SPDX-License-Identifier: LGPL-3.0-or-later
"""
pretrain_atomic_model.py  –  final fixed version
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple

import torch
from deepmd.dpmodel import FittingOutputDef,OutputVariableDef
from deepmd.pt.model.atomic_model.base_atomic_model import BaseAtomicModel
from deepmd.pt.model.atomic_model.dp_atomic_model import DPAtomicModel
from deepmd.pt.model.task.neighbor_fitting import NeighborCoordFitting
from deepmd.pt.model.task.fitting import Fitting

from typing import Iterable
def _as_list(vdefs) -> Iterable[OutputVariableDef]:
    """
    DeePMD <=2.1:   vdefs 是 list[OutputVariableDef]
    DeePMD  >=2.2:  vdefs 是 dict[str, OutputVariableDef]
    """
    if isinstance(vdefs, dict):
        return vdefs.values()
    return vdefs   # 旧版 list

def _merge_output_defs(*defs: FittingOutputDef) -> FittingOutputDef:
    merged: list[OutputVariableDef] = []
    for d in defs:
        # 有 .var_defs 属性的新接口
        if hasattr(d, "var_defs"):
            merged.extend(_as_list(d.var_defs))
        # 老接口：FittingOutputDef 本身就是 dict-like
        else:
            merged.extend(_as_list(d.values()))

    # 去重（按变量名保留第一次出现的定义）
    uniq: dict[str, OutputVariableDef] = {}
    for vv in merged:
        if vv.name not in uniq:
            uniq[vv.name] = vv
    return FittingOutputDef(list(uniq.values()))


@BaseAtomicModel.register("pretrain_descriptor")
class PretrainDescriptorAtomicModel(DPAtomicModel):
    """
    DPAtomicModel + Δr 回归头
    """

    def __init__(
        self,
        descriptor,
        fitting,
        type_map: list[str],
        nsel: Optional[int] = None,
        **kwargs,
    ) -> None:
        # ------------------------------------------------------------------ #
        # 0) 先放一个临时 output_def，供父类 __init__ 提前调用                 #
        # ------------------------------------------------------------------ #
        self._atomic_output_def = fitting.output_def()

        # ------------------------------------------------------------------ #
        # 1) 调用父类   (构建 descriptor / fitting_net / sel / …)            #
        # ------------------------------------------------------------------ #
        super().__init__(descriptor, fitting, type_map, **kwargs)

        # ------------------------------------------------------------------ #
        # 2) nsel 处理                                                       #
        # ------------------------------------------------------------------ #
        if nsel is None:
            nsel = sum(self.sel)
        self.nsel = int(nsel)

        # ------------------------------------------------------------------ #
        # 3) Δr 拟合头                                                       #
        # ------------------------------------------------------------------ #
       
        
        self.neighbor_fit: Fitting = NeighborCoordFitting(
           dim_pair=self.descriptor.get_dim_emb(),
            nsel=self.nsel,
        )

        # ------------------------------------------------------------------ #
        # 4) 生成最终 merged output_def                                       #
        # ------------------------------------------------------------------ #
       
        self._atomic_output_def = _merge_output_defs(
            self.fitting_net.output_def(),      # 能量 / 力
            self.neighbor_fit.output_def(),     # Δr
        )
    @torch.jit.export
    def fitting_output_def(self) -> FittingOutputDef:  # type: ignore[override]
        return self.atomic_output_def()
    # ---------------------------------------------------------------------- #
    #                    输出定义（两接口都要兼容）                           #
    # ---------------------------------------------------------------------- #
    @torch.jit.export
    def atomic_output_def(self) -> FittingOutputDef:  # noqa: D401
        """
        返回原有 atomic_output_def，并追加一个名为 ``mask`` 的输出变量。
        ``mask`` 形状为 [1]，不可约简，不参与梯度反传。
        """

        # 1. 先拿到“旧”定义（父类或 fitting_net 给出的）
        if hasattr(self, "_atomic_output_def"):
            base_def = self._atomic_output_def
        else:
            base_def = self.fitting_net.output_def()   # 最早期调用只能走这里

        # 2. 如果已经含有 "mask"，直接返回，避免重复创建
        if "mask" in base_def.keys():
            return base_def

        # 3. 把旧定义转成列表，然后 append 一个新的 OutputVariableDef
        var_list = list(base_def.get_data().values())
        var_list.append(
            OutputVariableDef(
                name="mask",
                shape=[1],
                reducible=False,           # 不在 frame 维度求和
                r_differentiable=False,    # 不需要对坐标求梯度
                c_differentiable=False,    # 不需要对元素类型求梯度
            )
        )

        # 4. 构造新的 FittingOutputDef，并缓存到 self 上，便于下次直接复用
        self._atomic_output_def = FittingOutputDef(var_list)
        print(self._atomic_output_def )
        return self._atomic_output_def

    
    def neighbor_triplet_cosine(
        self,
        nlist:  torch.Tensor,        # (bsz , natoms , nnei)   邻居索引，空位 = -1
        coord:  torch.Tensor,        # (bsz , nall   , 3)     扩展坐标 (含影像)
        eps: float = 1e-12           # 防 0 除
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算同一中心原子 i 的连续两邻居 (j_k , j_{k+1}) 与 i 构成的键角余弦:
            cosθ_k = cos( r_{j_k}-r_i , r_{j_{k+1}}-r_i )
            k = 0 … nnei-2

        返回
        ------
        cos_theta : (bsz , natoms , nnei-1)  float32/64
                    无效邻居对置 0
        mask_pair : (bsz , natoms , nnei-1)  bool
                    True → j_k 与 j_{k+1} 都存在
        """
        if nlist.dim() != 3:
            raise ValueError("nlist must be (bsz , natoms , nnei)")
        if nlist.size(2) < 2:
            raise ValueError("nnei must be ≥ 2 to form angles")

        bsz, natoms, nnei = nlist.shape
        device, dtype = coord.device, coord.dtype

        # -------- 1. 补哑原子，准备 int64 索引 ---------------------------------
        dummy = torch.zeros((bsz, 1, 3), dtype=dtype, device=device)
        coord = torch.cat([coord, dummy], dim=1)            # nall' = nall+1

        mask_atom  = nlist >= 0                             # True → 有邻居
        nlist_adj  = torch.where(mask_atom, nlist,
                                coord.size(1)-1).to(torch.long)  # (bsz,natoms,nnei)

        # -------- 2. gather 得到各邻居坐标 ------------------------------------
        idx   = nlist_adj.reshape(bsz, -1).unsqueeze(-1).expand(-1, -1, 3)
        coord_nei = torch.gather(coord, 1, idx)             # (bsz , natoms*nnei , 3)
        coord_nei = coord_nei.view(bsz, natoms, nnei, 3)    # (bsz , natoms , nnei , 3)

        # -------- 3. 中心原子坐标 (假设前 natoms 行即原胞本征原子) -------------
        coord_ctr = coord[:, :natoms, :].unsqueeze(2)       # (bsz , natoms , 1 , 3)
        coord_ctr = coord_ctr.expand(-1, -1, nnei, -1)      # (bsz , natoms , nnei , 3)

        # -------- 4. 构造向量 v_k = r_jk - r_i ---------------------------------
        vec = coord_nei - coord_ctr                         # (bsz,natoms,nnei,3)

        # -------- 5. 相邻两向量余弦 -------------------------------------------
        v1   = vec[:, :, :-1, :]                            # (bsz,natoms,nnei-1,3)
        v2   = vec[:, :, 1:,  :]                            # (bsz,natoms,nnei-1,3)

        mask_pair = mask_atom[:, :, :-1] & mask_atom[:, :, 1:]  # 有效角

        dot   = (v1 * v2).sum(dim=-1)                      # (bsz,natoms,nnei-1)
        n1    = v1.norm(dim=-1)
        n2    = v2.norm(dim=-1)
        cos_theta = dot / (n1 * n2 + eps)

        cos_theta = cos_theta.masked_fill(~mask_pair, 0.0)

       

        return cos_theta, mask_pair


    # ---------------------------------------------------------------------- #
    #                    前向传播：父类能量/力 + Δr                           #
    # ---------------------------------------------------------------------- #
    def forward_atomic(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        comm_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        # 1) 父类计算能量 / 力等
        
       
        ret = super().forward_atomic(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            comm_dict=comm_dict,
        )
        
      

        # 2) 描述符（父类虽然算过，但没存；重算一次损耗极小）
        
        
        # print(extended_atype)
        
        descrpt, rot_mat, g2, gt_rel_coord, sw = self.descriptor(
            extended_coord, extended_atype, nlist, mapping, comm_dict=comm_dict
        )

        gt_cos_theta, mask_trpl = self.neighbor_triplet_cosine(
            nlist, extended_coord
        )

        

       
        

        # 3) Δr 预测
        pred = self.neighbor_fit(g2,gt_rel_coord,nlist,gt_cos_theta)
        ret["rel_coord"] = pred["rel_coord"]
        ret["cos_theta"] = pred["cos_theta"]
       


        ret["gt_rel_coord"] = gt_rel_coord
        ret["gt_cos_theta"] = pred["gt_cos_theta"]


        
      

        # 4) 邻居 mask
        ret["neighbor_mask"] = nlist >= 0
        ret["triplet_mask"] = pred["triplet_mask"]

        # print(ret["triplet_mask"].shape)
        # print("hahahahha")
        # exit()

        

        return ret