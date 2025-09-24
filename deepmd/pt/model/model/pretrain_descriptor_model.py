# from typing import Optional

# import torch

# from deepmd.pt.model.atomic_model.pretrain_atomic_model import (
#     PretrainAtomicModel,
# )
# from deepmd.pt.model.model.model import (
#     BaseModel,
# )

# from .dp_model import (
#     DPModelCommon,
# )
# from .make_model import (
#     make_model,
# )

# # ----------------------------------------------------------------------
# # 先用工厂函数从原子模型生成框架模型
# # ----------------------------------------------------------------------
# DPPretrainModel_ = make_model(PretrainAtomicModel)


# @BaseModel.register("pretrain_descriptor")
# class PretrainDescriptorModel(DPModelCommon, DPPretrainModel_):
#     """Energy + Force + RelCoord 预测模型"""

#     model_type = "pretrain_descriptor"

#     # --------------------------------------------------------------
#     # 构造函数：直接复用父类
#     # --------------------------------------------------------------
#     def __init__(self, *args, **kwargs):
#         DPModelCommon.__init__(self)
#         DPPretrainModel_.__init__(self, *args, **kwargs)

#     # --------------------------------------------------------------
#     # 输出定义的翻译（给 protobuf / onnx 等用）
#     # --------------------------------------------------------------
#     def translated_output_def(self):
#         out_def_data = self.model_output_def().get_data()
#         output_def = {
#             "atom_energy": out_def_data["energy"],
#             "energy": out_def_data["energy_redu"],
#             "rel_coord": out_def_data["rel_coord"],
#         }
#         if self.do_grad_r("energy"):
#             output_def["force"] = out_def_data["energy_derv_r"]
#             output_def["force"].squeeze(-2)
#         if self.do_grad_c("energy"):
#             output_def["virial"] = out_def_data["energy_derv_c_redu"]
#             output_def["virial"].squeeze(-2)
#             output_def["atom_virial"] = out_def_data["energy_derv_c"]
#             output_def["atom_virial"].squeeze(-3)
#         if "mask" in out_def_data:
#             output_def["mask"] = out_def_data["mask"]
#         if "neighbor_mask" in out_def_data:
#             output_def["neighbor_mask"] = out_def_data["neighbor_mask"]
#         return output_def

#     # --------------------------------------------------------------
#     # 常规 forward（高层接口，自动建邻居 list）
#     # --------------------------------------------------------------
#     def forward(
#         self,
#         coord,
#         atype,
#         box: Optional[torch.Tensor] = None,
#         fparam: Optional[torch.Tensor] = None,
#         aparam: Optional[torch.Tensor] = None,
#         do_atomic_virial: bool = False,
#     ) -> dict[str, torch.Tensor]:

#         model_ret = self.forward_common(
#             coord,
#             atype,
#             box,
#             fparam=fparam,
#             aparam=aparam,
#             do_atomic_virial=do_atomic_virial,
#         )

#         # fitting_net 为空说明是 coordinate-denoise 任务，直接返回
#         if self.get_fitting_net() is None:
#             model_ret["updated_coord"] += coord
#             return model_ret

#         # 根据是否求导构造输出字典
#         model_predict: dict[str, torch.Tensor] = {
#             "atom_energy": model_ret["energy"],
#             "energy": model_ret["energy_redu"],
#             "rel_coord": model_ret["rel_coord"],
#         }

#         if self.do_grad_r("energy"):
#             model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
#         else:
#             model_predict["force"] = model_ret["dforce"]

#         if self.do_grad_c("energy"):
#             model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
#             if do_atomic_virial:
#                 model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(-3)

#         if "mask" in model_ret:
#             model_predict["mask"] = model_ret["mask"]
#         if "neighbor_mask" in model_ret:
#             model_predict["neighbor_mask"] = model_ret["neighbor_mask"]

#         return model_predict

#     # --------------------------------------------------------------
#     # forward_lower：用户自己提供邻居表
#     # --------------------------------------------------------------
#     @torch.jit.export
#     def forward_lower(
#         self,
#         extended_coord,
#         extended_atype,
#         nlist,
#         mapping: Optional[torch.Tensor] = None,
#         fparam: Optional[torch.Tensor] = None,
#         aparam: Optional[torch.Tensor] = None,
#         do_atomic_virial: bool = False,
#         comm_dict: Optional[dict[str, torch.Tensor]] = None,
#     ):
#         model_ret = self.forward_common_lower(
#             extended_coord,
#             extended_atype,
#             nlist,
#             mapping,
#             fparam=fparam,
#             aparam=aparam,
#             do_atomic_virial=do_atomic_virial,
#             comm_dict=comm_dict,
#             extra_nlist_sort=self.need_sorted_nlist_for_lower(),
#         )

#         if self.get_fitting_net() is None:
#             return model_ret  # denoise-only

#         model_predict: dict[str, torch.Tensor] = {
#             "atom_energy": model_ret["energy"],
#             "energy": model_ret["energy_redu"],
#             "rel_coord": model_ret["rel_coord"],
#         }

#         if self.do_grad_r("energy"):
#             model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
#         else:
#             assert model_ret["dforce"] is not None
#             model_predict["dforce"] = model_ret["dforce"]

#         if self.do_grad_c("energy"):
#             model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
#             if do_atomic_virial:
#                 model_predict["extended_virial"] = model_ret["energy_derv_c"].squeeze(
#                     -3
#                 )

#         if "neighbor_mask" in model_ret:
#             model_predict["neighbor_mask"] = model_ret["neighbor_mask"]

#         return model_predict

from typing import Optional

import torch

from deepmd.pt.model.atomic_model.pretrain_atomic_model import (
    PretrainDescriptorAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)

from .dp_model import (
    DPModelCommon,
)
from .make_model import (
    make_model,
)

# ----------------------------------------------------------------------
# 先用工厂函数从原子模型生成框架模型
# ----------------------------------------------------------------------
DPPretrainModel_ = make_model(PretrainDescriptorAtomicModel)


@BaseModel.register("pretrain_descriptor")
class PretrainDescriptorModel(DPModelCommon, DPPretrainModel_):
    """Energy + Force + RelCoord 预测模型"""

    model_type = "pretrain_descriptor"

    # --------------------------------------------------------------
    # 构造函数：直接复用父类
    # --------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        DPModelCommon.__init__(self)
        DPPretrainModel_.__init__(self, *args, **kwargs)

    # --------------------------------------------------------------
    # 输出定义的翻译（给 protobuf / onnx 等用）
    # --------------------------------------------------------------
    def translated_output_def(self):
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "atom_energy": out_def_data["energy"],
            "energy": out_def_data["energy_redu"],
            "rel_coord": out_def_data["rel_coord"],
            "gt_rel_coord": out_def_data["gt_rel_coord"],
            "neighbor_mask": out_def_data["neighbor_mask"]
        }
        if self.do_grad_r("energy"):
            output_def["force"] = out_def_data["energy_derv_r"]
            output_def["force"].squeeze(-2)
        if self.do_grad_c("energy"):
            output_def["virial"] = out_def_data["energy_derv_c_redu"]
            output_def["virial"].squeeze(-2)
            output_def["atom_virial"] = out_def_data["energy_derv_c"]
            output_def["atom_virial"].squeeze(-3)
        if "mask" in out_def_data:
            output_def["mask"] = out_def_data["mask"]
        if "neighbor_mask" in out_def_data:
            output_def["neighbor_mask"] = out_def_data["neighbor_mask"]
        return output_def

    # --------------------------------------------------------------
    # 常规 forward（高层接口，自动建邻居 list）
    # --------------------------------------------------------------
    def forward(
        self,
        coord,
        atype,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:

        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )

        # fitting_net 为空说明是 coordinate-denoise 任务，直接返回
        if self.get_fitting_net() is None:
            model_ret["updated_coord"] += coord
            return model_ret
        
        # 根据是否求导构造输出字典
        model_predict: dict[str, torch.Tensor] = {
            "atom_energy": model_ret["energy"],
            "energy": model_ret["energy_redu"],
            "rel_coord": model_ret["rel_coord"],
            "gt_rel_coord": model_ret["gt_rel_coord"],
            "neighbor_mask": model_ret["neighbor_mask"],
            "cos_theta": model_ret["cos_theta"],
            "gt_cos_theta": model_ret["gt_cos_theta"]
        }

        if self.do_grad_r("energy"):
            model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
        else:
            model_predict["force"] = model_ret["dforce"]

        if self.do_grad_c("energy"):
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(-3)

        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        if "neighbor_mask" in model_ret:
            model_predict["neighbor_mask"] = model_ret["neighbor_mask"]
        if "triplet_mask" in model_ret:
            model_predict["triplet_mask"] = model_ret["triplet_mask"]

        return model_predict

    # --------------------------------------------------------------
    # forward_lower：用户自己提供邻居表
    # --------------------------------------------------------------
    @torch.jit.export
    def forward_lower(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
    ):
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            comm_dict=comm_dict,
            extra_nlist_sort=self.need_sorted_nlist_for_lower(),
        )

        if self.get_fitting_net() is None:
            return model_ret  # denoise-only
       
        model_predict: dict[str, torch.Tensor] = {
            "atom_energy": model_ret["energy"],
            "energy": model_ret["energy_redu"],
            "rel_coord": model_ret["rel_coord"],
            "gt_rel_coord": model_ret["gt_rel_coord"],
            "neighbor_mask": model_ret["neighbor_mask"]
        }

        if self.do_grad_r("energy"):
            model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
        else:
            assert model_ret["dforce"] is not None
            model_predict["dforce"] = model_ret["dforce"]

        if self.do_grad_c("energy"):
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict["extended_virial"] = model_ret["energy_derv_c"].squeeze(
                    -3
                )

        if "neighbor_mask" in model_ret:
            model_predict["neighbor_mask"] = model_ret["neighbor_mask"]

        return model_predict