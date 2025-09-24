
# ASDP: Angular and Shell-Aware Deep Potential Energy Model for Molecular Dynamics
Official code implementation for ASDP.

## Installation

ASDP is implemented under the DeepMD-kit framework, for installation, please follow the [online documentation](https://docs.deepmodeling.com/projects/deepmd/en/stable/install/install-from-source.html) and install the pytorch version.


## Molecular System Dataset
In our experiments, 6 system datasets are trained and evaluated:
- [AlMgCu](https://www.aissquare.com/datasets/detail?name=AlMgCu_DPA_v1_0&id=139&pageType=datasets)
- [ANI-1](https://github.com/isayev/ANI1_dataset)
- [SSE-PBE](https://www.aissquare.com/datasets/detail?pageType=datasets&name=SSE-PBE_DPA_v1_0&id=146)
- [HECN](https://www.aissquare.com/datasets/detail?pageType=datasets&name=DP_HECN_cryst_liquid_dataset&id=321)
- [2D-In2Se3](https://www.aissquare.com/datasets/detail?name=In2Se3-2D-dpgen&id=7&pageType=datasets)
- [Organic reaction](https://www.aissquare.com/datasets/detail?pageType=datasets&name=Organic_reactions_dataset&id=218)

## Usage
For running the training and evaluation, run the command:
```
DP_ENABLE_TENSORFLOW=0 DP_ENABLE_PYTORCH=1 CUDA_VISIBLE_DEVICES=[GPU id] dp --pt train path/to/your/input/json
```
The json files of the 6 systems for ASDP can be found under the [ASDP_json_files](ASDP_json_files/) folder, and the training/testing splitting of each dataset is recorded in the json file.

One example of the json file is as follows:
```

{
    "_comment1": " model parameters",
    "model": {
        "type_map": [
            "C",
            "H",
            "O",
            "N"
        ],
        "descriptor": {
            "type": "se_atten",
            "sel": 15,
            "rcut_smth": 0.50,
            "rcut": 6.00,
            "neuron": [
                25,
                50,
                100
            ],
            "resnet_dt": false,
            "axis_neuron": 16,
            "seed": 1,
            "attn": 128,
            "attn_layer": 2,
            "attn_dotr": true,
            "attn_mask": false,
            "precision": "float64",
            "_comment2": " that's all",
            "k_map": {
                "0": 12,
                "1": 12,
                "2": 12,
                "3": 12
            }
        },
        "fitting_net": {
            "neuron": [
                240,
                240,
                240
            ],
            "resnet_dt": true,
            "precision": "float64",
            "seed": 1,
            "_comment3": " that's all"
        },
        "_comment4": " that's all"
    },
    "learning_rate": {
        "type": "exp",
        "decay_steps": 5000,
        "start_lr": 0.001,
        "stop_lr": 3.51e-6,
        "_comment5": "that's all"
    },
    "loss": {
        "type": "ener",
        "start_pref_e": 0.02,
        "limit_pref_e": 1,
        "start_pref_f": 1000,
        "limit_pref_f": 1,
        "start_pref_v": 0,
        "limit_pref_v": 0,
        "_comment6": " that's all"
    },
    "training": {
        "training_data": {
            "systems": [
				"/data/mixdata/train/10",
				"/data/mixdata/train/11",
				"/data/mixdata/train/12",
				"/data/mixdata/train/13",
				"/data/mixdata/train/14",
				"/data/mixdata/train/15",
				"/data/mixdata/train/5",
				"/data/mixdata/train/6",
				"/data/mixdata/train/7",
				"/data/mixdata/train/8",
				"/data/mixdata/train/9"
            ],
            "batch_size": 128,
            "_comment7": "that's all"
        },
        "validation_data": {
            "systems": [
				"/data/mixdata/test/16",
				"/data/mixdata/test/17",
				"/data/mixdata/test/18"
            ],
            "batch_size": 128,
            "numb_btch": 3,
            "_comment8": "that's all"
        },
        "train_component": "all",
        "numb_steps": 100000,
        "seed": 10,
        "disp_freq": 1000,
        "save_freq": 1000000,
        "_comment9": "that's all"
    },
    "_comment10": "that's all"
}
```
We explain some of the key fields:
- `model.type_map` : all element types that appear in the system dataset.
- `descriptor.type`: in this branch, set this field to be `se_atten` to run ASDP.
- `descriptor.sel`: number of selected neighbor for each atom.
-  `rcut`: cutoff radias.
-  `k_map`: the selection of the `s` values for each atom type.
-  `training.disp_freq`: the frequency of testing, the value is set to 1000, that means every 1000 steps, the trained model will be tested on the whole evaluation set.
