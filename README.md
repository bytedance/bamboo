# **B**yteDance **A**I **M**olecular Simulation **BOO**ster (BAMBOO)

Welcome to the repository of BAMBOO! This repository hosts the source code for creating a machine learning-based force field (MLFF) for molecular dynamics (MD) simulations of lithium battery electrolytes. Whether you're interested in simulating lithium battery electrolytes or other types of liquids, BAMBOO provides a robust and versatile solution.

Thank you for considering BAMBOO for your research needs. We are thrilled to be a part of your scientific journey and are eager to see how our project contributes to your outstanding results.

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

If you use BAMBOO in your research, please cite: [BAMBOO: a predictive and transferable machine learning force field framework for liquid electrolyte development](https://arxiv.org/abs/2404.07181).

## License

This project is licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).
