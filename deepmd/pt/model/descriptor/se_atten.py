# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    Optional,
    Union,
    Dict,
)

# from pyparsing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as torch_func

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.descriptor.descriptor import (
    DescriptorBlock,
)
from deepmd.pt.model.descriptor.env_mat import (
    prod_env_mat,
)
from deepmd.pt.model.network.layernorm import (
    LayerNorm,
)
from deepmd.pt.model.network.mlp import (
    EmbeddingNet,
    MLPLayer,
    NetworkCollection,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
)
from deepmd.pt.utils.env_mat_stat import (
    EnvMatStatSe,
)
from deepmd.pt.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.utils.env_mat_stat import (
    StatItem,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

if not hasattr(torch.ops.deepmd, "tabulate_fusion_se_atten"):

    def tabulate_fusion_se_atten(
        argument0: torch.Tensor,
        argument1: torch.Tensor,
        argument2: torch.Tensor,
        argument3: torch.Tensor,
        argument4: torch.Tensor,
        argument5: int,
        argument6: bool,
    ) -> list[torch.Tensor]:
        raise NotImplementedError(
            "tabulate_fusion_se_atten is not available since customized PyTorch OP library is not built when freezing the model. "
            "See documentation for model compression for details."
        )

    # Note: this hack cannot actually save a model that can be runned using LAMMPS.
    torch.ops.deepmd.tabulate_fusion_se_atten = tabulate_fusion_se_atten


@DescriptorBlock.register("se_atten")
class DescrptBlockSeAtten(DescriptorBlock):
    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: Union[list[int], int],
        ntypes: int,
        neuron: list = [25, 50, 100],
        axis_neuron: int = 16,
        tebd_dim: int = 8,
        tebd_input_mode: str = "concat",
        set_davg_zero: bool = True,
        attn: int = 128,
        attn_layer: int = 2,
        attn_dotr: bool = True,
        attn_mask: bool = False,
        activation_function="tanh",
        precision: str = "float64",
        resnet_dt: bool = False,
        scaling_factor=1.0,
        normalize=True,
        temperature=None,
        smooth: bool = True,
        type_one_side: bool = False,
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        trainable_ln: bool = True,
        ln_eps: Optional[float] = 1e-5,
        seed: Optional[Union[int, list[int]]] = None,
        type: Optional[str] = None,
        k_map: Optional[Dict[int, int]] = None,
    ) -> None:
        r"""Construct an embedding net of type `se_atten`.

        Parameters
        ----------
        rcut : float
            The cut-off radius :math:`r_c`
        rcut_smth : float
            From where the environment matrix should be smoothed :math:`r_s`
        sel : list[int], int
            list[int]: sel[i] specifies the maxmum number of type i atoms in the cut-off radius
            int: the total maxmum number of atoms in the cut-off radius
        ntypes : int
            Number of element types
        neuron : list[int]
            Number of neurons in each hidden layers of the embedding net :math:`\mathcal{N}`
        axis_neuron : int
            Number of the axis neuron :math:`M_2` (number of columns of the sub-matrix of the embedding matrix)
        tebd_dim : int
            Dimension of the type embedding
        tebd_input_mode : str
            The input mode of the type embedding. Supported modes are ["concat", "strip"].
            - "concat": Concatenate the type embedding with the smoothed radial information as the union input for the embedding network.
            - "strip": Use a separated embedding network for the type embedding and combine the output with the radial embedding network output.
        resnet_dt : bool
            Time-step `dt` in the resnet construction:
            y = x + dt * \phi (Wx + b)
        trainable_ln : bool
            Whether to use trainable shift and scale weights in layer normalization.
        ln_eps : float, Optional
            The epsilon value for layer normalization.
        type_one_side : bool
            If 'False', type embeddings of both neighbor and central atoms are considered.
            If 'True', only type embeddings of neighbor atoms are considered.
            Default is 'False'.
        attn : int
            Hidden dimension of the attention vectors
        attn_layer : int
            Number of attention layers
        attn_dotr : bool
            If dot the angular gate to the attention weights
        attn_mask : bool
            (Only support False to keep consistent with other backend references.)
            (Not used in this version.)
            If mask the diagonal of attention weights
        exclude_types : list[list[int]]
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
        env_protection : float
            Protection parameter to prevent division by zero errors during environment matrix calculations.
        set_davg_zero : bool
            Set the shift of embedding net input to zero.
        activation_function : str
            The activation function in the embedding net. Supported options are |ACTIVATION_FN|
        precision : str
            The precision of the embedding net parameters. Supported options are |PRECISION|
        scaling_factor : float
            The scaling factor of normalization in calculations of attention weights.
            If `temperature` is None, the scaling of attention weights is (N_dim * scaling_factor)**0.5
        normalize : bool
            Whether to normalize the hidden vectors in attention weights calculation.
        temperature : float
            If not None, the scaling of attention weights is `temperature` itself.
        seed : int, Optional
            Random seed for parameter initialization.
        """
        super().__init__()
        del type
        self.rcut = float(rcut)
        self.rcut_smth = float(rcut_smth)
        self.neuron = neuron
        self.filter_neuron = self.neuron
        self.axis_neuron = axis_neuron
        self.tebd_dim = tebd_dim
        self.tebd_input_mode = tebd_input_mode
        self.set_davg_zero = set_davg_zero
        self.attn_dim = attn
        self.attn_layer = attn_layer
        self.attn_dotr = attn_dotr
        self.attn_mask = attn_mask
        self.activation_function = activation_function
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
        self.resnet_dt = resnet_dt
        self.scaling_factor = scaling_factor
        self.normalize = normalize
        self.temperature = temperature
        self.smooth = smooth
        self.type_one_side = type_one_side
        self.env_protection = env_protection
        self.trainable_ln = trainable_ln
        self.seed = seed
        self.k_map = k_map
        #  to keep consistent with default value in this backends
        if ln_eps is None:
            ln_eps = 1e-5
        self.ln_eps = ln_eps

        if isinstance(sel, int):
            sel = [sel]

        self.ntypes = ntypes
        self.sel = sel
        self.sec = self.sel
        self.split_sel = self.sel
        self.nnei = sum(sel)
        self.ndescrpt = self.nnei * 4
        # order matters, placed after the assignment of self.ntypes
        self.reinit_exclude(exclude_types)

        self.dpa1_attention = NeighborGatedAttention(
            self.attn_layer,
            self.nnei,
            self.filter_neuron[-1],
            self.attn_dim,
            dotr=self.attn_dotr,
            do_mask=self.attn_mask,
            scaling_factor=self.scaling_factor,
            normalize=self.normalize,
            temperature=self.temperature,
            trainable_ln=self.trainable_ln,
            ln_eps=self.ln_eps,
            smooth=self.smooth,
            precision=self.precision,
            seed=child_seed(self.seed, 0),
            k_map=self.k_map,
        )

        wanted_shape = (self.ntypes, self.nnei, 4)
        mean = torch.zeros(wanted_shape, dtype=self.prec, device=env.DEVICE)
        stddev = torch.ones(wanted_shape, dtype=self.prec, device=env.DEVICE)
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        self.tebd_dim_input = self.tebd_dim if self.type_one_side else self.tebd_dim * 2
        if self.tebd_input_mode in ["concat"]:
            self.embd_input_dim = 1 + self.tebd_dim_input
        else:
            self.embd_input_dim = 1

        self.filter_layers_strip = None
        filter_layers = NetworkCollection(
            ndim=0, ntypes=self.ntypes, network_type="embedding_network"
        )
        filter_layers[0] = EmbeddingNet(
            self.embd_input_dim,
            self.filter_neuron,
            activation_function=self.activation_function,
            precision=self.precision,
            resnet_dt=self.resnet_dt,
            seed=child_seed(self.seed, 1),
        )
        self.filter_layers = filter_layers
        if self.tebd_input_mode in ["strip"]:
            filter_layers_strip = NetworkCollection(
                ndim=0, ntypes=self.ntypes, network_type="embedding_network"
            )
            filter_layers_strip[0] = EmbeddingNet(
                self.tebd_dim_input,
                self.filter_neuron,
                activation_function=self.activation_function,
                precision=self.precision,
                resnet_dt=self.resnet_dt,
                seed=child_seed(self.seed, 2),
            )
            self.filter_layers_strip = filter_layers_strip
        self.stats = None

        # add for compression
        self.compress = False
        self.is_sorted = False
        self.compress_info = nn.ParameterList(
            [nn.Parameter(torch.zeros(0, dtype=self.prec, device="cpu"))]
        )
        self.compress_data = nn.ParameterList(
            [nn.Parameter(torch.zeros(0, dtype=self.prec, device=env.DEVICE))]
        )

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.rcut

    def get_rcut_smth(self) -> float:
        """Returns the radius where the neighbor information starts to smoothly decay to 0."""
        return self.rcut_smth

    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        return sum(self.sel)

    def get_sel(self) -> list[int]:
        """Returns the number of selected atoms for each type."""
        return self.sel

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.ntypes

    def get_dim_in(self) -> int:
        """Returns the input dimension."""
        return self.dim_in

    def get_dim_out(self) -> int:
        """Returns the output dimension."""
        return self.dim_out

    def get_dim_rot_mat_1(self) -> int:
        """Returns the first dimension of the rotation matrix. The rotation is of shape dim_1 x 3."""
        return self.filter_neuron[-1]

    def get_dim_emb(self) -> int:
        """Returns the output dimension of embedding."""
        return self.filter_neuron[-1]

    def __setitem__(self, key, value) -> None:
        if key in ("avg", "data_avg", "davg"):
            self.mean = value
        elif key in ("std", "data_std", "dstd"):
            self.stddev = value
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        if key in ("avg", "data_avg", "davg"):
            return self.mean
        elif key in ("std", "data_std", "dstd"):
            return self.stddev
        else:
            raise KeyError(key)

    def mixed_types(self) -> bool:
        """If true, the descriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the descriptor
        1. assumes total number of atoms of each atom type aligned across frames;
        2. requires a neighbor list that distinguishes different atomic types.

        """
        return True

    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix."""
        return self.env_protection

    @property
    def dim_out(self):
        """Returns the output dimension of this descriptor."""
        return self.filter_neuron[-1] * self.axis_neuron

    @property
    def dim_in(self):
        """Returns the atomic input dimension of this descriptor."""
        return self.tebd_dim

    @property
    def dim_emb(self):
        """Returns the output dimension of embedding."""
        return self.get_dim_emb()

    def compute_input_stats(
        self,
        merged: Union[Callable[[], list[dict]], list[dict]],
        path: Optional[DPPath] = None,
    ) -> None:
        """
        Compute the input statistics (e.g. mean and stddev) for the descriptors from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        path : Optional[DPPath]
            The path to the stat file.

        """
        env_mat_stat = EnvMatStatSe(self)
        if path is not None:
            path = path / env_mat_stat.get_hash()
        if path is None or not path.is_dir():
            if callable(merged):
                # only get data for once
                sampled = merged()
            else:
                sampled = merged
        else:
            sampled = []
        env_mat_stat.load_or_compute_stats(sampled, path)
        self.stats = env_mat_stat.stats
        mean, stddev = env_mat_stat()
        if not self.set_davg_zero:
            self.mean.copy_(
                torch.tensor(mean, device=env.DEVICE, dtype=self.mean.dtype)
            )
        self.stddev.copy_(
            torch.tensor(stddev, device=env.DEVICE, dtype=self.stddev.dtype)
        )

    def get_stats(self) -> dict[str, StatItem]:
        """Get the statistics of the descriptor."""
        if self.stats is None:
            raise RuntimeError(
                "The statistics of the descriptor has not been computed."
            )
        return self.stats

    def reinit_exclude(
        self,
        exclude_types: list[tuple[int, int]] = [],
    ) -> None:
        self.exclude_types = exclude_types
        self.is_sorted = len(self.exclude_types) == 0
        self.emask = PairExcludeMask(self.ntypes, exclude_types=exclude_types)

    def enable_compression(
        self,
        table_data,
        table_config,
        lower,
        upper,
    ) -> None:
        net = "filter_net"
        self.compress_info[0] = torch.as_tensor(
            [
                lower[net],
                upper[net],
                upper[net] * table_config[0],
                table_config[1],
                table_config[2],
                table_config[3],
            ],
            dtype=self.prec,
            device="cpu",
        )
        self.compress_data[0] = table_data[net].to(device=env.DEVICE, dtype=self.prec)
        self.compress = True

    def forward(
        self,
        nlist: torch.Tensor,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        step: Optional[torch.Tensor] = None,
        extended_atype_embd: Optional[torch.Tensor] = None,
        mapping: Optional[torch.Tensor] = None,
        type_embedding: Optional[torch.Tensor] = None,
    ):
        """Compute the descriptor.

        Parameters
        ----------
        nlist
            The neighbor list. shape: nf x nloc x nnei
        extended_coord
            The extended coordinates of atoms. shape: nf x (nallx3)
        extended_atype
            The extended aotm types. shape: nf x nall x nt
        extended_atype_embd
            The extended type embedding of atoms. shape: nf x nall
        mapping
            The index mapping, not required by this descriptor.
        type_embedding
            Full type embeddings. shape: (ntypes+1) x nt
            Required for stripped type embeddings.

        Returns
        -------
        result
            The descriptor. shape: nf x nloc x (ng x axis_neuron)
        g2
            The rotationally invariant pair-partical representation.
            shape: nf x nloc x nnei x ng
        h2
            The rotationally equivariant pair-partical representation.
            shape: nf x nloc x nnei x 3
        gr
            The rotationally equivariant and permutationally invariant single particle
            representation. shape: nf x nloc x ng x 3
        sw
            The smooth switch function. shape: nf x nloc x nnei

        """
        
        del mapping
        assert extended_atype_embd is not None
        nframes, nloc, nnei = nlist.shape
        atype = extended_atype[:, :nloc] # Get the integer type for each atom
        atype_flat = atype.reshape(-1)

        nb = nframes
        nall = extended_coord.view(nb, -1, 3).shape[1]
        dmatrix, diff, sw = prod_env_mat(
            extended_coord,
            nlist,
            extended_atype[:, :nloc],
            self.mean,
            self.stddev,
            self.rcut,
            self.rcut_smth,
            protection=self.env_protection,
        )

       
        # nb x nloc x nnei
        exclude_mask = self.emask(nlist, extended_atype)
        
        nlist = torch.where(exclude_mask != 0, nlist, -1)
        nlist_mask = nlist != -1
        nlist = torch.where(nlist == -1, 0, nlist)
        sw = torch.squeeze(sw, -1)
        # nf x nall x nt
        nt = extended_atype_embd.shape[-1]
        # beyond the cutoff sw should be 0.0
        sw = sw.masked_fill(~nlist_mask, 0.0)
        # (nb x nloc) x nnei
        exclude_mask = exclude_mask.view(nb * nloc, nnei)
        # nfnl x nnei x 4
        dmatrix = dmatrix.view(-1, self.nnei, 4)
        nfnl = dmatrix.shape[0]
        # nfnl x nnei x 4
        rr = dmatrix
        rr = rr * exclude_mask[:, :, None]
        ss = rr[:, :, :1]
        if self.tebd_input_mode in ["concat"]:
            atype_tebd_ext = extended_atype_embd
            # nb x (nloc x nnei) x nt
            index = nlist.reshape(nb, nloc * nnei).unsqueeze(-1).expand(-1, -1, nt)
            # nb x (nloc x nnei) x nt
            atype_tebd_nlist = torch.gather(atype_tebd_ext, dim=1, index=index)  # j
            # nb x nloc x nnei x nt
            atype_tebd_nlist = atype_tebd_nlist.view(nb, nloc, nnei, nt)

            # nf x nloc x nt -> nf x nloc x nnei x nt
            atype_tebd = extended_atype_embd[:, :nloc, :]
            atype_tebd_nnei = atype_tebd.unsqueeze(2).expand(-1, -1, self.nnei, -1)  # i

            nlist_tebd = atype_tebd_nlist.reshape(nfnl, nnei, self.tebd_dim)
            atype_tebd = atype_tebd_nnei.reshape(nfnl, nnei, self.tebd_dim)
            if not self.type_one_side:
                # nfnl x nnei x (1 + tebd_dim * 2)
                ss = torch.concat([ss, nlist_tebd, atype_tebd], dim=2)
            else:
                # nfnl x nnei x (1 + tebd_dim)
                ss = torch.concat([ss, nlist_tebd], dim=2)
            # nfnl x nnei x ng
            gg = self.filter_layers.networks[0](ss)

           
            # dist = rr[..., 0]
            s = rr[..., 0]

            dist = 1 / rr[..., 0] + 1e-8

            '''
            # 获取每个中心原子实际的邻居数量
            nei_mask = dist  # nlist 中 -1 表示无邻居  != -1
            neighbor_counts = nei_mask.sum(dim=2)  # 统计每个中心原子的邻居数量，shape: [nf, nloc]

            # 将所有帧和所有中心原子的邻居数量拉平
            neighbor_counts_flat = neighbor_counts.view(-1)  # shape: [nf * nloc]

            # 统计 90% 中心原子的邻居数的最大值
            neighbor_counts_sorted, _ = torch.sort(neighbor_counts_flat)  # 从小到大排序
            ninety_percent_index = int(0.9 * len(neighbor_counts_sorted))  # 找到 90% 的索引
            ninety_percent_max_neighbors = neighbor_counts_sorted[ninety_percent_index].item()

            # 统计所有中心原子的最大邻居数量
            max_neighbors = neighbor_counts_flat.max().item()

            # 打印统计结果
            print(f"90% 中心原子的最大邻居数: {ninety_percent_max_neighbors}")
            print(f"所有中心原子的最大邻居数: {max_neighbors}")
            exit()
            '''
            '''
            print(atype.reshape(-1))  # 所有的原子eg【0， 1， 2， 3， 0， 1， 2， 1， 2】
            print(atype.reshape(-1).shape)  # shape

            print(dist)  # （所有）某个原子与所有邻居的距离矩阵
            print(dist.shape)

            import matplotlib.pyplot as plt
            import os
            import numpy as np
            import math

            plt.style.use("default")
            output_dir = "./distance_plots/HECN"
            os.makedirs(output_dir, exist_ok=True)

            # 转换数据到 CPU
            atype_np = atype.reshape(-1).cpu().numpy()   
            dist_np = dist.detach().cpu().numpy()        

            # 获取原子类型数量
            # unique_types = np.unique(atype_np)
            # plt.figure(figsize=(10, 6))
            # √
            # 可视化所有原子的邻居距离
            # N_atoms_to_plot = 2000
            # indices_to_plot = np.arange(min(N_atoms_to_plot, len(atype_np)))
            # √
            # 获取所有原子类型
            unique_types = np.unique(atype_np)
            type_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))  # 使用标准调色板
            type_color_map = {t: type_colors[i] for i, t in enumerate(unique_types)}

            # 自动排版：N 行 M 列
            n_types = len(unique_types)
            n_cols = 3
            n_rows = math.ceil(n_types / n_cols)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            axes = np.array(axes).flatten()  # 扁平化方便索引

            for idx, atype_id in enumerate(unique_types):
                ax = axes[idx]
                indices = np.where(atype_np == atype_id)[0]
                color = type_color_map[atype_id]

                for i in indices:
                    ax.plot(dist_np[i], color=color, alpha=0.4, linewidth=1.0, rasterized=True)

                ax.set_xlim(0, dist_np.shape[1] - 1)
                ax.set_ylim(0, 50.0)
                ax.set_title(f"Type {atype_id} ({len(indices)} atoms)")
                ax.grid(True, alpha=0.3)

            # 隐藏多余的子图
            for idx in range(n_types, len(axes)):
                fig.delaxes(axes[idx])


            # √
            # # 创建图形
            # plt.figure(figsize=(12, 7))
            # handles = {}  # 用于去重图例：记录每个 type 对应的 line handle

            # for i in indices_to_plot:
            #     atom_type = atype_np[i]
            #     color = type_color_map[atom_type]
            #     line, = plt.plot(
            #         dist_np[i],
            #         color=color,
            #         marker="o",
            #         markersize=3,
            #         linewidth=1.2,
            #         alpha=0.8,
            #         label=f"Type {atom_type}"
            #     )
            #     # 只保留第一个该类型的图例
            #     if atom_type not in handles:
            #         handles[atom_type] = line
            # plt.xlim(0, dist_np.shape[1] - 1)  # x: 从第0个到第nnei-1个邻居
            # √

            # plt.figure(figsize=(10, 6))
            # for i in range(N_atoms_to_plot):
            #     plt.plot(dist_np[i], label=f"Atom {i}", marker="o", markersize=3, linewidth=1)

            # for atype_id in unique_types:
            #     indices = np.where(atype_np == atype_id)[0]
            #     avg_distances = dist_np[indices].mean(axis=0)  # 对该类型所有原子取平均
            #     plt.plot(avg_distances, label=f"Type {atype_id}", linewidth=2)
            
            # plt.xlabel("Neighbor Index")
            # plt.ylabel("Distance (Å)")
            # plt.title("Neighbor Distances by Atom Type")
            # plt.grid(True, alpha=0.3)
            # plt.ylim(0, 2)  # 设置纵坐标只显示0到2的数据
            # plt.tight_layout()
            # plt.legend()
            # ax.show()
            # √
            ax.set_xlabel("Neighbor Index")
            ax.set_ylabel("Distance (Å)")
            ax.set_title(f"Type {atype_id} ({len(indices)} atoms)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            # ax.show()

            # 生成文件名并保存
            filename = f"system_distances_HECN_0_50.0.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved plot for System to {filepath}")
            
            exit()
            '''
            # thr = 0.25
            # delta = dist[..., :-1] - dist[..., 1:]   # shape (..., n-1)
            # pct   = delta.abs() / dist[..., :-1].abs()
            # mask  = pct > thr 
            # has   = mask.any(dim=-1)                   # 是否存在下跳
            # first = mask.float().argmax(dim=-1)        # 全 False 时得到 0
            # first = torch.where(
            #     has,
            #     first,
            #     dist.new_full(first.shape, dist.size(-1) - 1, dtype=torch.long)
            # )
            # torch.set_printoptions(threshold=torch.inf)
            # with open("dist.txt","w") as f:
            #     f.write(str(dist))

            # # Δ 位于 i ↔ i+1 之间，所以 k = first + 1
            # first_shell_K = first + 1                              # shape (...,)

            # print(first_shell_K)
            # print(first_shell_K.shape)
            input_r = torch.nn.functional.normalize(
                rr.reshape(-1, self.nnei, 4)[:, :, 1:4], dim=-1
            )

            atype_flat = atype.reshape(-1)
            
            gg = self.dpa1_attention(
                gg, nlist_mask, input_r=input_r, sw=sw, step=step, s=s, atype=atype_flat  
            )  # shape is [nframes*nloc, self.neei, out_size]
            # nfnl x 4 x ng  # 修改 加入atype=atype_flat  
            xyz_scatter = torch.matmul(rr.permute(0, 2, 1), gg)
        else:
            raise NotImplementedError

        xyz_scatter = xyz_scatter / self.nnei
        xyz_scatter_1 = xyz_scatter.permute(0, 2, 1)
        rot_mat = xyz_scatter_1[:, :, 1:4]
        xyz_scatter_2 = xyz_scatter[:, :, 0 : self.axis_neuron]
        result = torch.matmul(
            xyz_scatter_1, xyz_scatter_2
        )  # shape is [nframes*nloc, self.filter_neuron[-1], self.axis_neuron]
        
      
        return (
            result.view(nframes, nloc, self.filter_neuron[-1] * self.axis_neuron),
            gg.view(nframes, nloc, self.nnei, self.filter_neuron[-1])
            if not self.compress
            else None,
            dmatrix.view(nframes, nloc, self.nnei, 4)[..., 1:],
            rot_mat.view(nframes, nloc, self.filter_neuron[-1], 3),
            sw,
        )

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor block has message passing."""
        return False

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the descriptor block needs sorted nlist when using `forward_lower`."""
        return False


class NeighborGatedAttention(nn.Module):
    def __init__(
        self,
        layer_num: int,
        nnei: int,
        embed_dim: int,
        hidden_dim: int,
        dotr: bool = False,
        do_mask: bool = False,
        scaling_factor: float = 1.0,
        normalize: bool = True,
        temperature: Optional[float] = None,
        trainable_ln: bool = True,
        ln_eps: float = 1e-5,
        smooth: bool = True,
        precision: str = DEFAULT_PRECISION,
        seed: Optional[Union[int, list[int]]] = None,
        k_map: Optional[Dict[int, int]] = None, # MODIFICATION: Add k_map
    ) -> None:
        """Construct a neighbor-wise attention net."""
        super().__init__()
        self.layer_num = layer_num
        self.nnei = nnei
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dotr = dotr
        self.do_mask = do_mask
        self.scaling_factor = scaling_factor
        self.normalize = normalize
        self.temperature = temperature
        self.trainable_ln = trainable_ln
        self.ln_eps = ln_eps
        self.smooth = smooth
        self.precision = precision
        self.seed = seed
        self.k_map = k_map
        self.network_type = NeighborGatedAttentionLayer
        attention_layers = []
        for i in range(self.layer_num):
            attention_layers.append(
                NeighborGatedAttentionLayer(
                    nnei,
                    embed_dim,
                    hidden_dim,
                    dotr=dotr,
                    do_mask=do_mask,
                    scaling_factor=scaling_factor,
                    normalize=normalize,
                    temperature=temperature,
                    trainable_ln=trainable_ln,
                    ln_eps=ln_eps,
                    smooth=smooth,
                    precision=precision,
                    seed=child_seed(seed, i),
                    k_map=self.k_map,
                )
            )
        self.attention_layers = nn.ModuleList(attention_layers)

    def forward(
        self,
        input_G,
        nei_mask,
        input_r: Optional[torch.Tensor] = None,
        sw: Optional[torch.Tensor] = None,
        step: Optional[torch.Tensor] = None,
        s: Optional[torch.Tensor] = None,
        atype: Optional[torch.Tensor] = None,
    ):
        """Compute the multi-layer gated self-attention.

        Parameters
        ----------
        input_G
            inputs with shape: (nf x nloc) x nnei x embed_dim.
        nei_mask
            neighbor mask, with paddings being 0. shape: (nf x nloc) x nnei.
        input_r
            normalized radial. shape: (nf x nloc) x nnei x 3.
        sw
            The smooth switch function. shape: nf x nloc x nnei
        """
        out = input_G
        # https://github.com/pytorch/pytorch/issues/39165#issuecomment-635472592
        for layer in self.attention_layers:
            out = layer(out, nei_mask, input_r=input_r, sw=sw, step=step, s=s, atype=atype)
        return out

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.attention_layers[key]
        else:
            raise TypeError(key)

    def __setitem__(self, key, value) -> None:
        if not isinstance(key, int):
            raise TypeError(key)
        if isinstance(value, self.network_type):
            pass
        elif isinstance(value, dict):
            value = self.network_type.deserialize(value)
        else:
            raise TypeError(value)
        self.attention_layers[key] = value

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        return {
            "@class": "NeighborGatedAttention",
            "@version": 1,
            "layer_num": self.layer_num,
            "nnei": self.nnei,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "dotr": self.dotr,
            "do_mask": self.do_mask,
            "scaling_factor": self.scaling_factor,
            "normalize": self.normalize,
            "temperature": self.temperature,
            "trainable_ln": self.trainable_ln,
            "ln_eps": self.ln_eps,
            "precision": self.precision,
            "attention_layers": [layer.serialize() for layer in self.attention_layers],
        }

    @classmethod
    def deserialize(cls, data: dict) -> "NeighborGatedAttention":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        attention_layers = data.pop("attention_layers")
        obj = cls(**data)
        for ii, network in enumerate(attention_layers):
            obj[ii] = network
        return obj


class NeighborGatedAttentionLayer(nn.Module):
    def __init__(
        self,
        nnei: int,
        embed_dim: int,
        hidden_dim: int,
        dotr: bool = False,
        do_mask: bool = False,
        scaling_factor: float = 1.0,
        normalize: bool = True,
        temperature: Optional[float] = None,
        smooth: bool = True,
        trainable_ln: bool = True,
        ln_eps: float = 1e-5,
        precision: str = DEFAULT_PRECISION,
        seed: Optional[Union[int, list[int]]] = None,
        k_map: Optional[Dict[int, int]] = None,
    ) -> None:
        """Construct a neighbor-wise attention layer."""
        super().__init__()
        self.nnei = nnei
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dotr = dotr
        self.do_mask = do_mask
        self.scaling_factor = scaling_factor
        self.normalize = normalize
        self.temperature = temperature
        self.precision = precision
        self.trainable_ln = trainable_ln
        self.ln_eps = ln_eps
        self.seed = seed
        self.k_map = k_map
        self.attention_layer = GatedAttentionLayer(
            nnei,
            embed_dim,
            hidden_dim,
            dotr=dotr,
            do_mask=do_mask,
            scaling_factor=scaling_factor,
            normalize=normalize,
            temperature=temperature,
            smooth=smooth,
            precision=precision,
            seed=child_seed(seed, 0),
            k_map=self.k_map,
        )
        self.attn_layer_norm = LayerNorm(
            self.embed_dim,
            eps=ln_eps,
            trainable=trainable_ln,
            precision=precision,
            seed=child_seed(seed, 1),
        )

    def forward(
        self,
        x,
        nei_mask,
        input_r: Optional[torch.Tensor] = None,
        sw: Optional[torch.Tensor] = None,
        step: Optional[torch.Tensor] = None,
        s: Optional[torch.Tensor] = None,
        atype: Optional[torch.Tensor] = None,
    ):
        residual = x
        x, _ = self.attention_layer(x, nei_mask, input_r=input_r, sw=sw, step=step, s=s, atype=atype)
        x = residual + x
        x = self.attn_layer_norm(x)
        return x

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        return {
            "nnei": self.nnei,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "dotr": self.dotr,
            "do_mask": self.do_mask,
            "scaling_factor": self.scaling_factor,
            "normalize": self.normalize,
            "temperature": self.temperature,
            "trainable_ln": self.trainable_ln,
            "ln_eps": self.ln_eps,
            "precision": self.precision,
            "attention_layer": self.attention_layer.serialize(),
            "attn_layer_norm": self.attn_layer_norm.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "NeighborGatedAttentionLayer":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        attention_layer = data.pop("attention_layer")
        attn_layer_norm = data.pop("attn_layer_norm")
        obj = cls(**data)
        obj.attention_layer = GatedAttentionLayer.deserialize(attention_layer)
        obj.attn_layer_norm = LayerNorm.deserialize(attn_layer_norm)
        return obj


class GatedAttentionLayer(nn.Module):
    def __init__(
        self,
        nnei: int,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int = 1,         
        dotr: bool = False,
        do_mask: bool = False,
        scaling_factor: float = 1.0,
        normalize: bool = True,
        temperature: Optional[float] = None,
        bias: bool = True,
        smooth: bool = True,
        precision: str = DEFAULT_PRECISION,
        seed: Optional[Union[int, list[int]]] = None,
        use_angle_embedding: bool = True,  # 是否使用角度embedding
        angle_embedding_dim: int = 64,      # 角度embedding的隐藏维度
        k_map: Optional[Dict[int, int]] = None
    ) -> None:
        """Construct a multi-head neighbor-wise attention net."""
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.nnei = nnei
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dotr = dotr
        self.do_mask = do_mask
        self.bias = bias
        self.smooth = smooth
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        self.precision = precision
        self.prec = PRECISION_DICT[precision]
        self.seed = seed
        self.scaling = (
            (self.head_dim * scaling_factor) ** -0.5
            if temperature is None
            else temperature
        )
        self.normalize = normalize
        self.in_proj = MLPLayer(
            embed_dim,
            hidden_dim * 3,
            bias=bias,
            use_timestep=False,
            bavg=0.0,
            stddev=1.0,
            precision=precision,
            seed=child_seed(seed, 0),
        )
        self.out_proj = MLPLayer(
            hidden_dim,
            embed_dim,
            bias=bias,
            use_timestep=False,
            bavg=0.0,
            stddev=1.0,
            precision=precision,
            seed=child_seed(seed, 1),
        )

        self.use_angle_embedding = use_angle_embedding
        if k_map:
            self.k_map = {int(k): v for k, v in k_map.items()}
            print(f"--- [VERIFY-GATED-ATTENTION-INIT] Received k_map: {k_map} ---")
        else:
            # Provide a default if k_map is not given in the input
            self.k_map = {} 
            print("--- [DeePMD-kit WARNING] k_map not provided, angular bias will not be used. ---")
            self.use_angle_embedding = False

        num_angle_features = 6
        if self.use_angle_embedding:
            # # MODIFICATION: Define k_map, with a default if not provided
            # self.k_map = {
            #     0 : 32,
            #     1 : 32
            # }

            self.angle_encoder = nn.Sequential(
                nn.Linear(num_angle_features, angle_embedding_dim),
                nn.SiLU(),
                nn.Linear(angle_embedding_dim, num_heads)
            ).to(self.prec)
            # 可选：层归一化
            self.angle_norm = nn.LayerNorm(num_heads).to(self.prec)
            # 可选：可学习的缩放因子
            self.angle_scale = nn.Parameter(torch.ones(1, dtype=PRECISION_DICT[precision]))
            
    def _compute_angles(self, input_r,s_k: Optional[torch.Tensor] = None):
        """
        计算所有原子对之间的角度
        
        Parameters
        ----------
        input_r : torch.Tensor
            归一化的径向向量，shape: (nf x nloc) x nnei x 3
            
        Returns
        -------
        angles : torch.Tensor
            角度矩阵，shape: (nf x nloc) x nnei x nnei
            范围在 [0, π]
        """
        # 计算余弦相似度
        # (nf x nloc) x nnei x nnei
       
        cos_angles = torch.matmul(input_r, input_r.transpose(-2, -1))
        
        # 裁剪到 [-1, 1] 范围内（数值稳定性）
        cos_angles = torch.clamp(cos_angles, -1.0, 1.0).to(self.prec)

        sin_angles = torch.sqrt(1 - cos_angles**2 + 1e-8)

        sin_2theta = 2 * sin_angles * cos_angles

        kappa = 2.0
        von_mises = torch.exp(kappa * (cos_angles - 1))
        
        angle_features = torch.stack([cos_angles, sin_angles, sin_2theta, von_mises], dim=-1)

        if s_k is not None:
            batch_size, nnei = s_k.shape
            s_j_matrix = s_k.unsqueeze(2).expand(-1, -1, nnei)
            s_k_matrix = s_k.unsqueeze(1).expand(-1, nnei, -1)
            s_sum = s_j_matrix + s_k_matrix
            s_diff = torch.abs(s_j_matrix - s_k_matrix)
            s_features = torch.stack([s_sum, s_diff], dim=-1)
            angle_features = torch.cat([angle_features, s_features], dim=-1)
        
        # print(angle_features.shape)
        # exit()
        
        return angle_features


        
        # # 计算角度（弧度）
        # angles = torch.acos(cos_angles)
        # angles = angles.to(self.precision)
    
        # return angles
    def forward(
        self,
        query,
        nei_mask,
        input_r: Optional[torch.Tensor] = None,
        sw: Optional[torch.Tensor] = None,
        attnw_shift: float = 20.0,
        step: Optional[torch.Tensor] = None,
        s: Optional[torch.Tensor] = None,
        atype: Optional[torch.Tensor] = None,
    ):
        """Compute the multi-head gated self-attention.

        Parameters
        ----------
        query
            inputs with shape: (nf x nloc) x nnei x embed_dim.
        nei_mask
            neighbor mask, with paddings being 0. shape: (nf x nloc) x nnei.
        input_r
            normalized radial. shape: (nf x nloc) x nnei x 3.
        sw
            The smooth switch function. shape: (nf x nloc) x nnei
        attnw_shift : float
            The attention weight shift to preserve smoothness when doing padding before softmax.
        """
        
        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        # Reshape for multi-head attention: (nf x nloc) x num_heads x nnei x head_dim
        q = q.view(-1, self.nnei, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(-1, self.nnei, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(-1, self.nnei, self.num_heads, self.head_dim).transpose(1, 2)

        if self.normalize:
            q = torch_func.normalize(q, dim=-1)
            k = torch_func.normalize(k, dim=-1)
            v = torch_func.normalize(v, dim=-1)

        q = q * self.scaling
        # (nf x nloc) x num_heads x head_dim x nnei
        k = k.transpose(-2, -1)

        # Compute attention scores
        # (nf x nloc) x num_heads x nnei x nnei
        attn_weights = torch.matmul(q, k)

        # MODIFICATION: Replaced the flawed loop with a vectorized, atype-based approach.
        if self.use_angle_embedding and input_r is not None and atype is not None and s is not None:
            n_atoms, n_nei, _ = input_r.shape
    
            # --- 步骤 1: 计算 k_vals 和 K_max (与之前相同) ---
            map_keys = torch.tensor(list(self.k_map.keys()), device=atype.device, dtype=torch.long)
            map_values = torch.tensor(list(self.k_map.values()), device=atype.device, dtype=torch.long)
            max_idx = max(atype.max(), map_keys.max())
            lookup_table = torch.full((max_idx + 1,), n_nei, device=atype.device, dtype=torch.long)
            lookup_table[map_keys] = map_values
            k_vals = lookup_table[atype]

            # --- 步骤 2: 【优化核心】找到当前批次所需的最大 K 值 ---
            # .item() 将单元素张量从 GPU 传回 CPU，这是一个廉价操作，因为只传输一个数字
            K_max = k_vals.max().item()

            # --- 步骤 3: 【优化核心】仅对前 K_max 个邻居计算角度特征 ---
            # 仅切片需要计算的部分，大大减少了 _compute_angles 的计算量
            input_r_k_max = input_r[:, :K_max, :]
            s_k_max = s[:, :K_max]
            
            # 在切片后的小张量上执行昂贵的计算
            angle_features_k_max = self._compute_angles(input_r_k_max, s_k=s_k_max)
            
           
            # --- 步骤 4: 【优化核心】在小张量上进行编码，然后填充回完整尺寸 ---
            # 编码器现在处理的是 (n_atoms, K_max, K_max, feature_dim) 的小张量
            angle_bias_k_max = self.angle_encoder(angle_features_k_max)
            angle_bias_k_max = self.angle_norm(angle_bias_k_max)
            angle_bias_k_max = angle_bias_k_max * self.angle_scale

            # 创建一个正确尺寸的全零张量，准备接收计算结果
            num_heads = angle_bias_k_max.shape[-1]  # 从计算结果中获取头的数量
            angle_bias = torch.zeros(
                n_atoms, n_nei, n_nei, num_heads,
                device=input_r.device,
                dtype=angle_bias_k_max.dtype
            )

            # 将计算好的 K_max x K_max 的偏置块 "填充" 到全零张量的左上角
            angle_bias[:, :K_max, :K_max, :] = angle_bias_k_max
            
            # --- 步骤 5: 应用精确的 k_mask (与之前相同，但现在至关重要) ---
            # 创建每个原子精确的 K 值掩码
            k_vals_unsqueezed = k_vals.unsqueeze(1)
            nei_range = torch.arange(n_nei, device=input_r.device).expand(n_atoms, -1)
            
            k_mask = (nei_range < k_vals_unsqueezed)

            # 将一维掩码扩展为二维，用于邻居对
            k_mask_2d = k_mask.unsqueeze(2) & k_mask.unsqueeze(1)

           
            
            # 使用这个精确的掩码将填充区域中多余的计算结果清零
            # 这一步是保证正确性的关键！
            angle_bias = angle_bias.masked_fill(~k_mask_2d.unsqueeze(-1), 0.0)

            # --- 步骤 6: 添加偏置到注意力权重 (与之前相同) ---
            angle_bias = angle_bias.permute(0, 3, 1, 2)

            '''
            K = 18  #原来的
            DECAY_START_STEP = 50000
            DECAY_END_STEP = 150000
            if self.use_angle_embedding and input_r is not None:
                # origin
                # input_r_k = input_r[:, :K, :]
                # s_k = s[:, :K]
                # angles_k = self._compute_angles(input_r_k, s_k=s_k)
                # 原来的
            '''
        


            # √
            # # 计算角度
            # angles_k = self._compute_angles(input_r_k, s_k=s_k)
            # # 将角度编码为bias
            # angle_bias_k = self.angle_encoder(angles_k)  
            
            # # 归一化（可选）
            # angle_bias_k = self.angle_norm(angle_bias_k)
            
            # # 缩放
            # angle_bias_k = angle_bias_k * self.angle_scale
            
            # # 获取维度信息
            # batch, nnei = input_r.shape[0], input_r.shape[1]
            # num_heads = angle_bias_k.shape[-1]  # 获取heads数量
            
            # # 创建完整的角度偏置矩阵（注意：4维）
            # angle_bias = torch.zeros(
            #     batch, nnei, nnei, num_heads,  # 添加num_heads维度
            #     dtype=angle_bias_k.dtype, 
            #     device=angle_bias_k.device
            # )
            
            # # 填充左上角K×K区域
            # # 原来的
            # # angle_bias[:, :K, :K, :] = angle_bias_k   # shape(angle_bias_k)=(batch, K, K, num_heads)
            # angle_bias[:, :K, :K, :] = angle_bias_k 
            # √

            # 转换为注意力机制需要的形状
            # (batch, nnei, nnei, num_heads) -> (batch, num_heads, nnei, nnei)
            

            decay_factor = 1.0
            # if step is not None:
            #     # 从 tensor 中获取 step 的数值
            #     step_val = step.item()

            #     # 3. 根据 step 应用 cosine decay 逻辑
            #     if step_val > DECAY_START_STEP:
            #         if step_val >= DECAY_END_STEP:
            #             # 超过结束点，衰减因子直接为 0
            #             decay_factor = 0.0
            #         else:
            #             # 在衰减区间内，计算 cosine decay
            #             # progress: 在衰减区间的进度 (从 0 到 1)
            #             progress = (step_val - DECAY_START_STEP) / (DECAY_END_STEP - DECAY_START_STEP)
            #             # Cosine decay 公式: 0.5 * (1 + cos(pi * progress))
            #             # 当 progress=0, cos(0)=1, factor=1
            #             # 当 progress=1, cos(pi)=-1, factor=0
            #             decay_factor = 0.5 * (1 + torch.cos(torch.tensor(torch.pi * progress, device=angle_bias.device)))
            #             # 转换回 python float，或者保持为 tensor 也可以
            #             decay_factor = decay_factor.item()

            # print("step",step, "decay_factor",decay_factor)

            angle_bias = angle_bias * decay_factor
            # 加性调节（而不是乘性）
          
            attn_weights = attn_weights + angle_bias


        # (nf x nloc) x nnei
        nei_mask = nei_mask.view(-1, self.nnei)

        if self.smooth:
            assert sw is not None
            # (nf x nloc) x 1 x nnei
            sw = sw.view(-1, 1, self.nnei)
            attn_weights = (attn_weights + attnw_shift) * sw[:, :, :, None] * sw[
                :, :, None, :
            ] - attnw_shift
        else:
            # (nf x nloc) x 1 x 1 x nnei
            attn_weights = attn_weights.masked_fill(
                ~nei_mask.unsqueeze(1).unsqueeze(1), float("-inf")
            )

        attn_weights = torch_func.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.masked_fill(
            ~nei_mask.unsqueeze(1).unsqueeze(-1), 0.0
        )
        if self.smooth:
            assert sw is not None
            attn_weights = attn_weights * sw[:, :, :, None] * sw[:, :, None, :]

       
        # if self.dotr:
        #     # (nf x nloc) x nnei x 3
        #     assert input_r is not None, "input_r must be provided when dotr is True!"
        #     # (nf x nloc) x 1 x nnei x nnei
        #     angular_weight = torch.matmul(input_r, input_r.transpose(-2, -1)).view(
        #         -1, 1, self.nnei, self.nnei
        #     )
        #     attn_weights = attn_weights * angular_weight

        # Apply attention to values
        # (nf x nloc) x nnei x (num_heads x head_dim)
        o = (
            torch.matmul(attn_weights, v)
            .transpose(1, 2)
            .reshape(-1, self.nnei, self.hidden_dim)
        )
        output = self.out_proj(o)
        return output, attn_weights

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        return {
            "nnei": self.nnei,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "dotr": self.dotr,
            "do_mask": self.do_mask,
            "scaling_factor": self.scaling_factor,
            "normalize": self.normalize,
            "temperature": self.temperature,
            "bias": self.bias,
            "smooth": self.smooth,
            "precision": self.precision,
            "in_proj": self.in_proj.serialize(),
            "out_proj": self.out_proj.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "GatedAttentionLayer":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        in_proj = data.pop("in_proj")
        out_proj = data.pop("out_proj")
        obj = cls(**data)
        obj.in_proj = MLPLayer.deserialize(in_proj)
        obj.out_proj = MLPLayer.deserialize(out_proj)
        return obj
