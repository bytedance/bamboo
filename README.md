# **B**yteDance **A**I **M**olecular Simulation **BOO**ster (BAMBOO)

Welcome to the repository of BAMBOO! This repository hosts the source code for creating a machine learning-based force field (MLFF) for molecular dynamics (MD) simulations of lithium battery electrolytes. Whether you're interested in simulating lithium battery electrolytes or other types of liquids, BAMBOO provides a robust and versatile solution.

Thank you for considering BAMBOO for your research needs. We are thrilled to be a part of your scientific journey and are eager to see how our project contributes to your outstanding results.

## 2025.05 Release
In this release, we provide the following updates about data, checkpoints, and implementation of dispersion corrections:

### Dataset
The complete training and validation DFT dataset is provided at: https://huggingface.co/datasets/mzl/bamboo

### Implementation of dispersion correction 
In the previous release, dispersion correction was constructed as element-dependent but independent of geometry. In this new release, we provide a new implementation of dispersion correction that depends on the coordination number (CN) of the atoms \[[2](#ref2), [3](#ref3)\]. This new implementation follows the original spirit of the DFT-D3(CSO) dispersion correction, with the value $a_4$ slightly adjusted for better zero-shot density prediction (see comment in `models/modules/dftd3/dftd3.py`).

### Checkpoints
In this release, we also provide the model checkpoint with the old implementation of dispersion correction that can reproduce the results presented in the paper [[1]](#ref1), as well as the model checkpoint with the newly implemented dispersion correction, at `benchmark/paper_new_disp.pt`.  These two checkpoints have similar performance in terms of prediction of density, viscosity, and ionic conductivity. Some benchmarks of the checkpoint with the newly implemented dispersion correction are listed below:

| System | Predicted Density (g/ml) | Exp. Density (g/ml) | Predicted Viscosity (cP) | Exp. Viscosity (cP) | Predicted Conductivity by Mistry (mS/cm) | Predicted Conductivity by NE (mS/cm) | Exp. Conductivity(mS/cm) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DEC | 0.971 +- 0.003 | 0.97 | 0.769 +- 0.020 | 0.749 |
| DMC | 1.053 +- 0.003 | 1.06 | 0.573 +- 0.011 | 0.585 |
| EA | 0.904 +- 0.004 | 0.9 | 0.536 +- 0.013 | 0.43 |
| EC | 1.328 +- 0.003 | 1.32 | 1.492 +- 0.021 | 1.93 |
| FEC | 1.499 +- 0.002 | 1.477 | 2.295 +- 0.031 | 2.24-4.1 |
| PC | 1.201 +- 0.003 | 1.2 | 1.755 +- 0.034 | 2.53 |
| Novec7000 | 1.399 +- 0.007 | 1.4 | 0.480 +- 0.008 | 0.45 |
| DMC_EC\|60_40\|LiPF6\|0.9 | 1.238 +- 0.003 | 1.239 | 2.942 +- 0.138 | 2.778 | 10.593 +- 0.512 | 13.560 +- 0.744 | 12.793 |
| DMC_EC_EMC\|45_50_5\|LiPF6\|1.1 | 1.273 +- 0.004 | 1.273 | 4.207 +- 0.190 | 4.566 | 8.061 +- 0.392 | 10.596 +- 0.571 | 12.175 |
| DMC_EC\|70_30\|LiPF6\|1.50 | 1.260 +- 0.003 | 1.268 | 4.916 +- 0.281 | 4.472 | 7.761 +- 0.282 | 10.795 +- 0.355 | 12.144 |
| DMC\|LiFSI\|2.22 | 1.276 +- 0.003 | 1.27 | 5.406 +- 0.545 | 3.9 | 8.642 +- 0.489 | 13.338 +- 0.494 | 12.2 |
| EC\|LiFSI\|0.49 | 1.375 +- 0.004 | 1.38 | 3.462 +- 0.129 | 4.1 | 5.913 +- 0.193 | 6.759 +- 0.241 | 8.7 |
| EC\|LiFSI\|1.14 | 1.419 +- 0.003 | 1.43 | 7.072 +- 0.333 | 8.1 | 4.724 +- 0.279 | 6.070 +- 0.383 | 9.7 |
| EMC | 1.018 +- 0.002 | 1 | 0.720 +- 0.009 | 0.65 |
| VC | 1.326 +- 0.004 | 1.355 | 1.013 +- 0.010 | 1.78 |
| ACT | 0.808 +- 0.001 | 0.79 | 0.469 +- 0.011 | 0.31 |
| DMC\|LiFSI\|3.70 | 1.361 +- 0.003 | 1.36 | 15.443 +- 2.009 | 12.9 | 3.504 +- 0.311 | 5.524 +- 0.337 | 8.1 |
| DMC\|LiFSI\|1.11 | 1.167 +- 0.001 | 1.18 | 1.903 +- 0.049 | 1.5 | 15.359 +- 0.504 | 20.113 +- 0.787 | 9.9 |
| EC\|LiFSI\|3.78 | 1.555 +- 0.002 | 1.57 | 38.690 +- 7.094 | 0 | 1.434 +- 0.187 | 2.064 +- 0.164 | 2.3 |
| EC\|LiFSI\|2.27 | 1.484 +- 0.001 | 1.5 | 18.041 +- 2.852 | 33.1 | 2.340 +- 0.354 | 3.138 +- 0.499 | 5.6 |
| DMC_EC\|51_49\|LiFSI\|3.74 | 1.456 +- 0.002 | 1.46 | 27.497 +- 3.017 | 36.9 | 1.899 +- 0.153 | 2.772 +- 0.173 | 4.5 |
| DMC_EC\|51_49\|LiFSI\|2.25 | 1.372 +- 0.002 | 1.38 | 10.376 +- 0.685 | 9.8 | 4.124 +- 0.214 | 5.889 +- 0.285 | 9.8 |
| DMC_EC\|51_49\|LiFSI\|1.12 | 1.285 +- 0.002 | 1.3 | 3.777 +- 0.096 | 3.4 | 9.296 +- 0.114 | 12.336 +- 0.155 | 14 |
| DMC_EC\|50_50\|LiPF6\|0.5 | 1.225 +- 0.002 | 1.235 | 2.120 +- 0.080 | 2.219 | 9.506 +- 0.453 | 11.011 +- 0.572 | 12.1420533 |
| DMC_EC\|70_30\|LiPF6\|1.00 | 1.219 +- 0.003 | 1.228 | 2.877 +- 0.122 | 2.734 | 11.260 +- 0.506 | 14.655 +- 0.693 | 12.4265334 |
| DMC_EC\|70_30\|LiPF6\|1.30 | 1.244 +- 0.002 | 1.252 | 4.111 +- 0.202 | 3.767 | 8.858 +- 0.378 | 12.098 +- 0.543 | 12.117719 |
| MA | 1.027 +- 0.002 | 0.93 | 0.552 +- 0.018 | 0.36 |

### References

<a id="ref1">\[1\]</a> Gong, Sheng, et al. "A predictive machine learning force-field framework for liquid electrolyte development." Nature Machine Intelligence (2026): 1-10. \
<a id="ref2">\[2\]</a> Grimme, Stefan, et al. "A consistent and accurate ab initio parametrization of density functional dispersion correction (DFT-D) for the 94 elements H-Pu." The Journal of chemical physics 132.15 (2010). \
<a id="ref3">\[3\]</a> Schröder, Heiner, Anne Creon, and Tobias Schwabe. "Reformulation of the D3 (Becke–Johnson) dispersion correction without resorting to higher than C 6 dispersion coefficients." Journal of chemical theory and computation 11.7 (2015): 3163-3170.


## Getting Started

This section will guide you on how to obtain and set up BAMBOO on your local machine for development and testing purposes.

### Prerequisites

To get started with BAMBOO, please ensure that you meet the following requirements:

- LAMMPS: stable_2Aug2023_update3 (Tested branch.)
- CUDA: 12+
- Pytorch: 2.0+

Once you have satisfied the above prerequisites, you are ready to proceed to the installation steps.

### Installing

To get started, clone the BAMBOO repository to your local machine using the following command:

```bash
git clone https://github.com/bytedance/bamboo.git
```

With this step, you get BAMBOO on your local system, ready for use.

To initialize the environment and retrieve the LAMMPS source code, follow these steps:

```bash
cd pair
bash ./init_compile.sh
cd lammps
bash ./build.sh
```

> The build.sh script is pre-configured for the NVIDIA GeForce RTX 4090 GPU. If you are using a different GPU, you may need to adjust the ARCH variable within the script to match your specific hardware. Refer to the NVIDIA CUDA Toolkit documentation for details on selecting the correct architecture flags.

> The Libtorch version is currently specified in the init_compile.sh script. If you require a different version of Libtorch, you will need to update this script accordingly.


## User Manual

To demonstrate the capabilities and usage of BAMBOO, we have included a small but self-contained dataset featuring key components used in electrolyte for lithium batteries. This dataset includes:

- **Dimethyl carbonate (DMC)**
- **Ethylene carbonate (EC)**
- **Lithium ions (Li<sup>+</sup>)**
- **Hexafluorophosphate ions (PF<sub>6</sub><sup>-</sup>)**

To get the dataset, you need:

1. Visit the following links to download the datasets: [Demo data](https://huggingface.co/datasets/mzl/bamboo)

2. After downloading, copy the `train_data.pt` and `val_data.pt` into the `data` directory of the project. Once the datasets are properly placed, you can proceed with the following examples. 

As we focus on simulating an electrolyte composed of DMC, EC, and LiPF<sub>6</sub>, we also provide:

- **Initial conformation file**: `in.data` in folder `benchmark`, which contains the starting structure for MD simulations.
- **Input file for LAMMPS**: `in.lammps` in folder `benchmark`, which is prepared to start simulations using LAMMPS.

These resources are designed to help users quickly set up BAMBOO and run simulations based on MLFF to explore the behavior of lithium battery electrolytes.

### Train a MLFF Model

Follow these steps to train a MLFF using BAMBOO:

1. **Navigate to the project directory**

   Replace `<path-to-your-installation>` with the actual path where you have installed BAMBOO, then execute the following command to move into that directory:

   ```bash
   cd <path-to-your-installation>
   ```
2. **Train a model**

   Start the training process by running:

   ```bash
   python3 -m train.train --config configs/train_config/config.json
   ```

   This command uses a configuration file located at `configs/train_config/config.json`, where the paremeters can be changed as you need. After training, a new folder named after the `job_name` variable in your configuration file will be created inside the `<path-to-your-installation>/train` directory. This folder will contain the training logs and checkpoint models saved as `.pt` files.

### Run a MD Simulation using a BAMBOO MLFF Model

To perform a MD simulation using a BAMBOO model, follow these steps:

1. **Create a folder for MD simulation and prepare the necessary files**

   Navigate to your BAMBOO directory and make a new folder for MD simulations. Copy the `in.data` and `in.lammps` files from `<path-to-your-installation>/data` into this directory:

   ```bash
   cd <path-to-your-installation> 
   mkdir simulation && cd simulation 
   cp ../benchmark/* .
   ```
2. **Configure the simulation settings**

   Modify the `benchmark.pt` in `in.lammps` file to point to the path of `.pt` file for the simulation.
   
3. **Run a MD simulation**

   Execute a MD simulation by LAMMPS:

   ```bash
   lmp -k on g 1 -sf kk -in in.lammps -log log.lammps > out.log 2>&1
   ```

   The `in.lammps` file can be configured for your simulation needs. The `.pt` file from any MLFF generated from training, ensembling, or alignment, can be used to run the MD simulations.

### Generate Frames for Ensemble and Alignment

To run ensemble and alignment processes, frames from MD trajectories are required. Here's a guide to generating these frames:

1. **Navigate to the project directory**

   Execute the following command to move into that directory:

   ```bash
   cd <path-to-your-installation>
   ```
2. **Extract the frames from MD trajectories**

   Here is an example command to extract frames from MD trajectories:

   ```bash
   python3 -m utils.load_traj --job_folder <path-to-your-simulation> --output_folder <path-to-save-frames> --mixture_name <your-mixture-name>
   ```

   The mixture-name will be used in the alignment to instruct which system is aligned.

### Ensemble a model

Averaging multiple replicate MLFF models into an ensembled one can help reduce variance and improve the accuracy of predictions. Follow these steps to ensemble several models trained from your dataset:

1. **Navigate to the project directory**

   Execute the following command to move into that directory:

   ```bash
   cd <path-to-your-installation>
   ```
2. **Modify the config file**

   To ensemble your models, you need to modify the `config.json` file appropriately. This file should clearly define the paths to the models you intend to ensemble, the model based on which the changes of paremeters will be made, and the directories containing the MD frames used for ensembling. Here, we give an example of `config.json`.

   ```json
   {
    "job_name": "ensemble_bamboo_community",
    "training_data_path": "<path-to-your-installation>/data/train_data.pt",
    "validation_data_path": "<path-to-your-installation>/data/val_data.pt",
    "batch_size": 512,
    "models": ["<path-to-your-model>/<your-model1-name>.pt", "<path-to-your-model>/<your-model2-name>.pt", "<path-to-your-model>/<your-model3-name>.pt"], 
    "frame_directories": ["<path-to-your-frames>"],
    "ensemble_model": "<path-to-your-model>/<your-ensemble-model>.pt",
    "validation_split_ratio": 0.1,
    "lr": 1e-6,
    "epochs": 50,
    "scheduler_gamma": 0.99,
    "validation_interval": 10,
    "energy_ratio": 0.3,
    "force_ratio": 1.0,
    "virial_ratio": 0.1,
    "bulk_energy_ratio": 0.01,
    "bulk_force_ratio": 3.0,
    "bulk_virial_ratio": 0.01,
    "max_frames_per_mixture": 960,
    "frame_validation_interval": 3
   }
   ```
   In this file, the `models` is a list containing all the paths of models you intend to ensemble. The `frame_direcories` is a list containing all the paths of MD frames used. The `ensemble_model` is the path of the based-model, whose parameters will change.
3. **Ensemble the models**

   Start the ensemble process by running:

   ```bash
   python3 -m train.ensemble --config configs/ensemble_config/config.json
   ```

   After ensembling, a new folder named after the `job_name` variable in your configuration file will be created inside the `<path-to-your-installation>/ensemble` directory. This folder will contain the training logs and checkpoint models saved as `.pt` files.

**Note**: To create an ensemble model, you need at least three different models.

### Alignment

BAMBOO offers functionality to finetune the model's predictions by adjusting parameters such as pressure, which is referred to as the alignment process. For example, if you need to change the model's predicted pressure by dP = -2000 Pa, follow these specific steps:

1. **Navigate to the project directory**

   Execute the following command to move into that directory:

   ```bash
   cd <path-to-your-installation>
   ```
2. **Modify the config file**

   To finetune your models by the alignment, you need to modify the `config.json` file appropriately. This file should clearly define the paths to the model you intend to finetune, and the directories containing the MD frames used for alignment. Here, we give an example of `config.json`.

   ```json
   {
    "job_name": "alignment_bamboo_community",
    "training_data_path": "<path-to-your-installation>/data/train_data.pt",
    "validation_data_path": "<path-to-your-installation>/data/val_data.pt",
    "model": "<path-to-your-model>/<your-alignment-model>.pt", 
    "frame_directories": ["<path-to-your-frames>"],
    "mixture_names": ["<your-mixture-name>"],
    "delta_pressure": [-2000],
    "energy_ratio": 0.3,
    "force_ratio": 1.0, 
    "virial_ratio": 0.1,
    "dipole_ratio": 3.0,
    "bulk_energy_ratio": 1e2,
    "bulk_force_ratio": 1e6,
    "bulk_virial_ratio": 3e3,
    "batch_size": 512,
    "epochs": 30,
    "frame_val_interval": 3,
    "max_frame_per_mixture": 30,
    "lr": 1e-12,
    "scheduler_gamma": 0.99
   }
   ```
   The `mixture_names` is a list that includes the names of the mixtures corresponding to the frames, which is set during generating frames. The `delta_pressure` is a list that contains the values of dP for each mixture.
3. **Finetune the model by the alignment process**

   Start the alignment process by running:

   ```bash
   python3 -m train.alignment --config configs/alignment_config/config.json
   ```

   After alignment, a new folder named after the `job_name` variable in your configuration file will be created inside the `<path-to-your-installation>/alignment` directory. This folder will contain the training logs and checkpoint models saved as `.pt` files.

## Benchmark Model
We have provided the model we trained that used for the data reported in our paper, which is located in the `benchmark` folder and named `benchmark.pt`. If you wish to reproduce the results mentioned in the paper, you can use this model. 

## Contributing

We welcome contributions to BAMBOO! If you have suggestions or improvements, please refers to `CONTRIBUTING.md`

## Citing BAMBOO

If you use BAMBOO in your research, please cite:

```bibtex
@article{gong_predictive_2025,
	title = {A predictive machine learning force-field framework for liquid electrolyte development},
	volume = {7},
	issn = {2522-5839},
	url = {https://doi.org/10.1038/s42256-025-01009-7},
	doi = {10.1038/s42256-025-01009-7},
	number = {4},
	journal = {Nature Machine Intelligence},
	author = {Gong, Sheng and Zhang, Yumin and Mu, Zhenliang and Pu, Zhichen and Wang, Hongyi and Han, Xu and Yu, Zhiao and Chen, Mengyi and Zheng, Tianze and Wang, Zhi and Chen, Lifei and Yang, Zhenze and Wu, Xiaojie and Shi, Shaochen and Gao, Weihao and Yan, Wen and Xiang, Liang},
	month = apr,
	year = {2025},
	pages = {543--552},
}
```

## License

This project is licensed under the [GNU General Public License, Version 2](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html).
