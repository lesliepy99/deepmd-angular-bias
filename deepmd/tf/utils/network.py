# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np

from deepmd.tf.common import (
    get_precision,
)
from deepmd.tf.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    tf,
)


def one_layer_rand_seed_shift() -> int:
    return 3


def one_layer(
    inputs,
    outputs_size,
    activation_fn=tf.nn.tanh,
    precision=GLOBAL_TF_FLOAT_PRECISION,
    stddev=1.0,
    bavg=0.0,
    name="linear",
    scope="",
    reuse=None,
    seed=None,
    use_timestep=False,
    trainable=True,
    useBN=False,
    uniform_seed=False,
    initial_variables=None,
    mixed_prec=None,
    final_layer=False,
):
    # For good accuracy, the last layer of the fitting network uses a higher precision neuron network.
    if mixed_prec is not None and final_layer:
        inputs = tf.cast(inputs, get_precision(mixed_prec["output_prec"]))
    with tf.variable_scope(name, reuse=reuse):
        shape = inputs.get_shape().as_list()
        w_initializer = tf.random_normal_initializer(
            stddev=stddev / np.sqrt(shape[1] + outputs_size),
            seed=seed if (seed is None or uniform_seed) else seed + 0,
        )
        b_initializer = tf.random_normal_initializer(
            stddev=stddev,
            mean=bavg,
            seed=seed if (seed is None or uniform_seed) else seed + 1,
        )
        if initial_variables is not None:
            w_initializer = tf.constant_initializer(
                initial_variables[scope + name + "/matrix"]
            )
            b_initializer = tf.constant_initializer(
                initial_variables[scope + name + "/bias"]
            )
        w = tf.get_variable(
            "matrix",
            [shape[1], outputs_size],
            precision,
            w_initializer,
            trainable=trainable,
        )
        variable_summaries(w, "matrix")
        b = tf.get_variable(
            "bias", [outputs_size], precision, b_initializer, trainable=trainable
        )
        variable_summaries(b, "bias")

        if mixed_prec is not None and not final_layer:
            inputs = tf.cast(inputs, get_precision(mixed_prec["compute_prec"]))
            w = tf.cast(w, get_precision(mixed_prec["compute_prec"]))
            b = tf.cast(b, get_precision(mixed_prec["compute_prec"]))

        hidden = tf.nn.bias_add(tf.matmul(inputs, w), b)
        if activation_fn is not None and use_timestep:
            idt_initializer = tf.random_normal_initializer(
                stddev=0.001,
                mean=0.1,
                seed=seed if (seed is None or uniform_seed) else seed + 2,
            )
            if initial_variables is not None:
                idt_initializer = tf.constant_initializer(
                    initial_variables[scope + name + "/idt"]
                )
            idt = tf.get_variable(
                "idt", [outputs_size], precision, idt_initializer, trainable=trainable
            )
            variable_summaries(idt, "idt")
        if activation_fn is not None:
            if useBN:
                None
                # hidden_bn = self._batch_norm(hidden, name=name+'_normalization', reuse=reuse)
                # return activation_fn(hidden_bn)
            else:
                if use_timestep:
                    if mixed_prec is not None and not final_layer:
                        idt = tf.cast(idt, get_precision(mixed_prec["compute_prec"]))
                    hidden = tf.reshape(activation_fn(hidden), [-1, outputs_size]) * idt
                else:
                    hidden = tf.reshape(activation_fn(hidden), [-1, outputs_size])

        if mixed_prec is not None:
            hidden = tf.cast(hidden, get_precision(mixed_prec["output_prec"]))
        return hidden


def layer_norm_tf(x, shape, weight=None, bias=None, eps=1e-5):
    """
    Layer normalization implementation in TensorFlow.

    Parameters
    ----------
    x : tf.Tensor
        The input tensor.
    shape : tuple
        The shape of the weight and bias tensors.
    weight : tf.Tensor
        The weight tensor.
    bias : tf.Tensor
        The bias tensor.
    eps : float
        A small value added to prevent division by zero.

    Returns
    -------
    tf.Tensor
        The normalized output tensor.
    """
    # Calculate the mean and variance
    mean = tf.reduce_mean(x, axis=list(range(-len(shape), 0)), keepdims=True)
    variance = tf.reduce_mean(
        tf.square(x - mean), axis=list(range(-len(shape), 0)), keepdims=True
    )

    # Normalize the input
    x_ln = (x - mean) / tf.sqrt(variance + eps)

    # Scale and shift the normalized input
    if weight is not None and bias is not None:
        x_ln = x_ln * weight + bias

    return x_ln


def layernorm(
    inputs,
    outputs_size,
    precision=GLOBAL_TF_FLOAT_PRECISION,
    name="linear",
    scope="",
    reuse=None,
    seed=None,
    uniform_seed=False,
    uni_init=True,
    eps=1e-5,
    trainable=True,
    initial_variables=None,
):
    with tf.variable_scope(name, reuse=reuse):
        shape = inputs.get_shape().as_list()
        if uni_init:
            gamma_initializer = tf.ones_initializer()
            beta_initializer = tf.zeros_initializer()
        else:
            gamma_initializer = tf.random_normal_initializer(
                seed=seed if (seed is None or uniform_seed) else seed + 0
            )
            beta_initializer = tf.random_normal_initializer(
                seed=seed if (seed is None or uniform_seed) else seed + 1
            )
        if initial_variables is not None:
            gamma_initializer = tf.constant_initializer(
                initial_variables[scope + name + "/gamma"]
            )
            beta_initializer = tf.constant_initializer(
                initial_variables[scope + name + "/beta"]
            )
        gamma = tf.get_variable(
            "gamma",
            [outputs_size],
            precision,
            gamma_initializer,
            trainable=trainable,
        )
        variable_summaries(gamma, "gamma")
        beta = tf.get_variable(
            "beta", [outputs_size], precision, beta_initializer, trainable=trainable
        )
        variable_summaries(beta, "beta")

        output = layer_norm_tf(
            inputs,
            (outputs_size,),
            weight=gamma,
            bias=beta,
            eps=eps,
        )
        return output


def embedding_net_rand_seed_shift(network_size):
    shift = 3 * (len(network_size) + 1)
    return shift


def embedding_net(
    xx,
    network_size,
    precision,
    activation_fn=tf.nn.tanh,
    resnet_dt=False,
    name_suffix="",
    stddev=1.0,
    bavg=0.0,
    seed=None,
    trainable=True,
    uniform_seed=False,
    initial_variables=None,
    mixed_prec=None,
    bias=True,
):
    r"""The embedding network.

    The embedding network function :math:`\mathcal{N}` is constructed by is the
    composition of multiple layers :math:`\mathcal{L}^{(i)}`:

    .. math::
        \mathcal{N} = \mathcal{L}^{(n)} \circ \mathcal{L}^{(n-1)}
        \circ \cdots \circ \mathcal{L}^{(1)}

    A layer :math:`\mathcal{L}` is given by one of the following forms,
    depending on the number of nodes: [1]_

    .. math::
        \mathbf{y}=\mathcal{L}(\mathbf{x};\mathbf{w},\mathbf{b})=
        \begin{cases}
            \boldsymbol{\phi}(\mathbf{x}^T\mathbf{w}+\mathbf{b}) + \mathbf{x}, & N_2=N_1 \\
            \boldsymbol{\phi}(\mathbf{x}^T\mathbf{w}+\mathbf{b}) + (\mathbf{x}, \mathbf{x}), & N_2 = 2N_1\\
            \boldsymbol{\phi}(\mathbf{x}^T\mathbf{w}+\mathbf{b}), & \text{otherwise} \\
        \end{cases}

    where :math:`\mathbf{x} \in \mathbb{R}^{N_1}` is the input vector and :math:`\mathbf{y} \in \mathbb{R}^{N_2}`
    is the output vector. :math:`\mathbf{w} \in \mathbb{R}^{N_1 \times N_2}` and
    :math:`\mathbf{b} \in \mathbb{R}^{N_2}` are weights and biases, respectively,
    both of which are trainable if `trainable` is `True`. :math:`\boldsymbol{\phi}`
    is the activation function.

    Parameters
    ----------
    xx : Tensor
        Input tensor :math:`\mathbf{x}` of shape [-1,1]
    network_size : list of int
        Size of the embedding network. For example [16,32,64]
    precision:
        Precision of network weights. For example, tf.float64
    activation_fn:
        Activation function :math:`\boldsymbol{\phi}`
    resnet_dt : boolean
        Using time-step in the ResNet construction
    name_suffix : str
        The name suffix append to each variable.
    stddev : float
        Standard deviation of initializing network parameters
    bavg : float
        Mean of network initial bias
    seed : int
        Random seed for initializing network parameters
    trainable : boolean
        If the network is trainable
    uniform_seed : boolean
        Only for the purpose of backward compatibility, retrieves the old behavior of using the random seed
    initial_variables : dict
        The input dict which stores the embedding net variables
    mixed_prec
        The input dict which stores the mixed precision setting for the embedding net
    bias : bool, Optional
        Whether to use bias in the embedding layer.

    References
    ----------
    .. [1] Kaiming  He,  Xiangyu  Zhang,  Shaoqing  Ren,  and  Jian  Sun. Identitymappings
       in deep residual networks. InComputer Vision - ECCV 2016,pages 630-645. Springer
       International Publishing, 2016.
    """
    input_shape = xx.get_shape().as_list()
    outputs_size = [input_shape[1], *network_size]

    for ii in range(1, len(outputs_size)):
        w_initializer = tf.random_normal_initializer(
            stddev=stddev / np.sqrt(outputs_size[ii] + outputs_size[ii - 1]),
            seed=seed if (seed is None or uniform_seed) else seed + ii * 3 + 0,
        )
        b_initializer = (
            tf.random_normal_initializer(
                stddev=stddev,
                mean=bavg,
                seed=seed if (seed is None or uniform_seed) else seed + 3 * ii + 1,
            )
            if bias
            else None
        )
        if initial_variables is not None:
            scope = tf.get_variable_scope().name
            w_initializer = tf.constant_initializer(
                initial_variables[scope + "/matrix_" + str(ii) + name_suffix]
            )
            bias = (scope + "/bias_" + str(ii) + name_suffix) in initial_variables
            b_initializer = (
                tf.constant_initializer(
                    initial_variables[scope + "/bias_" + str(ii) + name_suffix]
                )
                if bias
                else None
            )
        w = tf.get_variable(
            "matrix_" + str(ii) + name_suffix,
            [outputs_size[ii - 1], outputs_size[ii]],
            precision,
            w_initializer,
            trainable=trainable,
        )
        variable_summaries(w, "matrix_" + str(ii) + name_suffix)

        b = (
            tf.get_variable(
                "bias_" + str(ii) + name_suffix,
                [outputs_size[ii]],
                precision,
                b_initializer,
                trainable=trainable,
            )
            if bias
            else None
        )
        if bias:
            variable_summaries(b, "bias_" + str(ii) + name_suffix)

        if mixed_prec is not None:
            xx = tf.cast(xx, get_precision(mixed_prec["compute_prec"]))
            w = tf.cast(w, get_precision(mixed_prec["compute_prec"]))
            b = tf.cast(b, get_precision(mixed_prec["compute_prec"])) if bias else None
        if activation_fn is not None:
            hidden = tf.reshape(
                activation_fn(
                    tf.nn.bias_add(tf.matmul(xx, w), b) if bias else tf.matmul(xx, w)
                ),
                [-1, outputs_size[ii]],
            )
        else:
            hidden = tf.reshape(
                tf.nn.bias_add(tf.matmul(xx, w), b) if bias else tf.matmul(xx, w),
                [-1, outputs_size[ii]],
            )
        if resnet_dt:
            idt_initializer = tf.random_normal_initializer(
                stddev=0.001,
                mean=1.0,
                seed=seed if (seed is None or uniform_seed) else seed + 3 * ii + 2,
            )
            if initial_variables is not None:
                scope = tf.get_variable_scope().name
                idt_initializer = tf.constant_initializer(
                    initial_variables[scope + "/idt_" + str(ii) + name_suffix]
                )
            idt = tf.get_variable(
                "idt_" + str(ii) + name_suffix,
                [1, outputs_size[ii]],
                precision,
                idt_initializer,
                trainable=trainable,
            )
            variable_summaries(idt, "idt_" + str(ii) + name_suffix)
            if mixed_prec is not None:
                idt = tf.cast(idt, get_precision(mixed_prec["compute_prec"]))

        if outputs_size[ii] == outputs_size[ii - 1]:
            if resnet_dt:
                xx += hidden * idt
            else:
                xx += hidden
        elif outputs_size[ii] == outputs_size[ii - 1] * 2:
            if resnet_dt:
                xx = tf.concat([xx, xx], 1) + hidden * idt
            else:
                xx = tf.concat([xx, xx], 1) + hidden
        else:
            xx = hidden
    if mixed_prec is not None:
        xx = tf.cast(xx, get_precision(mixed_prec["output_prec"]))
    return xx


def variable_summaries(var: tf.Variable, name: str) -> None:
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization).

    Parameters
    ----------
    var : tf.Variable
        [description]
    name : str
        variable name
    """
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)

        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        tf.summary.histogram("histogram", var)

def add_gumbel_noise(logits,
                     step,
                     noise_std_init=1.0,
                     noise_decay_steps=20_000):
    """
    Add Gumbel noise to `logits` for Noisy-Top-K routing.

    Parameters
    ----------
    logits : tf.Tensor
        The raw (pre-softmax) logits, shape = [..., N].
    step : tf.Tensor or int
        Global training step.
    noise_std_init : float
        Initial noise strength σ₀. 0 → never add noise.
    noise_decay_steps : int
        Linear-decay length. 0 → **do NOT add any noise at all**.

    Returns
    -------
    tf.Tensor
        `logits` with (optional) Gumbel noise.
    """

    # --- 0. 快速短路：完全不加噪声 --------------------------
    if noise_std_init == 0.0 or noise_decay_steps == 0:
        # 直接返回原 logits，不创建任何随机 op
        return logits

    # --------------------------------------------------------
    dtype = logits.dtype                        # 与 logits 保持一致
    step_f = tf.cast(step, dtype)
    decay_steps_f = tf.cast(noise_decay_steps, dtype)

    # 1. 线性衰减系数：σ(t) = σ₀ · max(1 − t / decay_steps, 0)
    progress   = step_f / decay_steps_f         # 0 → 1
    noise_std  = noise_std_init * tf.maximum(1.0 - progress, 0.0)

    # 2. 采样 Gumbel(0,1) 噪声
    uniform = tf.random.uniform(tf.shape(logits),
                                minval=1e-6, maxval=1.0,
                                dtype=dtype)
    gumbel  = -tf.math.log(-tf.math.log(uniform))   # G = -log(-log U)

    # 3. 注入噪声
    return logits + gumbel * noise_std

def topk_moe_block(
    inputs,
    num_experts: int,
    common_experts: int,
    expert_neurons: list[int],
    k: int = 2,
    gating_neurons: list[int] = None,
    activation_fn=tf.nn.relu,
    precision=GLOBAL_TF_FLOAT_PRECISION,
    stddev: float = 1.0,
    bavg: float = 0.0,
    use_timestep: bool = False,
    trainable: bool = True,
    uniform_seed: bool = False,
    seed: int = None,
    initial_variables: dict = None,
    mixed_prec: dict = None,
    name: str = "TopKMoEBlock",
    reuse=None,
    eps: float = 1e-5,        # layer-norm eps
    temp: float = 2.0,         # NEW softmax temperature
    step: int =0,
    is_training: bool = False,
    noise_gating_steps: int=20000
):
    """Top-k MoE with c common experts + (N−c) specific experts."""
    with tf.variable_scope(name, reuse=reuse):
        batch = tf.shape(inputs)[0]
        hdim   = inputs.get_shape().as_list()[-1]
       
     
        expert_dim = expert_neurons[-1]

        # 1) LayerNorm 先规范化输入，然后送进 gating 子网
        g = layernorm(
                inputs,
                outputs_size=hdim,
                precision=precision,
                name="gating_input_ln",     
                scope=name+"/",
                reuse=reuse,
                seed=seed,
                uniform_seed=uniform_seed,
                eps=eps,
                trainable=trainable,
                initial_variables=initial_variables,
        )  # shape = [B, hdim]

        # 1) gating 网络，只产出 N−c 个 logits
        if gating_neurons:
            for i, size in enumerate(gating_neurons):
                g = one_layer(
                    g, size, activation_fn=activation_fn,
                    precision=precision, stddev=stddev, bavg=bavg,
                    name=f"gating_h{i}", scope=name+"/",
                    reuse=reuse, seed=seed,
                    use_timestep=use_timestep,
                    trainable=trainable,
                    uniform_seed=uniform_seed,
                    initial_variables=initial_variables,
                    mixed_prec=mixed_prec,
                )
       
        logits_spec = one_layer(
            g, num_experts - common_experts,
            activation_fn=None,
            precision=precision, stddev=stddev, bavg=bavg,
            name="gating_logits_spec", scope=name+"/",
            reuse=reuse, seed=seed,
            use_timestep=False,
            trainable=trainable,
            uniform_seed=uniform_seed,
            initial_variables=initial_variables,
            mixed_prec=mixed_prec,
        )  # [batch, N−c]
        
        # 对 logits 做 token-wise 标准化，再 softmax( /T )
        mu  = tf.reduce_mean(logits_spec, axis=-1, keepdims=True)
        var = tf.reduce_mean(tf.square(logits_spec - mu), axis=-1, keepdims=True)
        logits_std  = tf.sqrt(var + 1e-5)
        logits_norm = (logits_spec - mu) / logits_std          # 0-均值 1-方差

        logits_for_topk = tf.cond(
            is_training,
            lambda: add_gumbel_noise(logits_norm,step,noise_decay_steps=noise_gating_steps),   # 训练：加噪声
            lambda: logits_norm                      # 推理：不用噪声
        )
        gating_probs = tf.nn.softmax(logits_for_topk  / temp, axis=-1)  # [B, N-c]

        # gating_probs = tf.nn.softmax(logits_spec)

        # 2) 构造所有 expert 并 stack
        expert_outputs = []
        for ei in range(num_experts):
            e = inputs
            for li, hsize in enumerate(expert_neurons):
                e = one_layer(
                    e, hsize, activation_fn=activation_fn,
                    precision=precision, stddev=stddev, bavg=bavg,
                    name=f"expert{ei}_h{li}", scope=name+"/",
                    reuse=reuse,
                    seed=(seed + ei*100) if seed is not None else None,
                    use_timestep=use_timestep,
                    trainable=trainable,
                    uniform_seed=uniform_seed,
                    initial_variables=initial_variables,
                    mixed_prec=mixed_prec,
                )
            expert_outputs.append(e)  # [batch, expert_dim]
        expert_stack = tf.stack(expert_outputs, axis=1)  # [batch, N, expert_dim]

        # 3) split common vs specific stacks
        if common_experts > 0:
            common_stack   = expert_stack[:, :common_experts, :]   # [batch, c, dim]
            specific_stack = expert_stack[:, common_experts:, :]   # [batch, N−c, dim]
        else:
            common_stack   = tf.zeros([batch, 0, expert_dim], dtype=precision)
            specific_stack = expert_stack                            # [batch, N, dim]

        # 4) 在 gating_probs ([batch, N−c]) 上做 Top-k
        # k_dynamic = tf.cond(
        #     step < 2000,
        #     lambda: tf.constant(num_experts - common_experts, dtype=tf.int32),
        #     lambda: tf.constant(k, dtype=tf.int32)
        # )
        k_dynamic =k

        topk_vals, topk_idx = tf.nn.top_k(gating_probs, k=k_dynamic, sorted=False)
        mask_spec = tf.reduce_sum(
            tf.one_hot(topk_idx, depth=(num_experts-common_experts), dtype=precision),
            axis=1
        )  # [batch, N−c], 0/1

        # 5) mask + renormalize
        gated_spec = gating_probs * mask_spec  # [batch, N−c]
        # denom = tf.reduce_sum(gated_spec, axis=-1, keepdims=True) + 1e-9
        # gated_spec = gated_spec / denom       # [batch, N−c], sum=1

        # 6) 计算输出
        # 6.1 common experts 直接 weight=1 求和
        if common_experts > 0:
            out_common = tf.reduce_sum(common_stack, axis=1)     # [batch, dim]
        else:
            out_common = tf.zeros([batch, expert_dim], dtype=precision)

        # 6.2 specific experts 用 gated_spec 加权
        out_spec = tf.reduce_sum(
            tf.expand_dims(gated_spec, -1) * specific_stack,
            axis=1
        )  # [batch, dim]
        # print(out_spec)
        
        # output = out_common + out_spec  # [batch, dim]
        output = tf.concat([out_common + out_spec,g],axis=-1) # concatenating gate network outputs and experts output
        # print(output)
        # exit()
        return output, gating_probs, mask_spec