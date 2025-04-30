#!/bin/bash

# Initialize Conda for this script
eval "$(conda shell.bash hook)"

# Define environment name and Python version
ENV_NAME=tce
PYTHON_VERSION=3.10

# Create a new conda environment
echo "Creating a new conda environment named $ENV_NAME with Python $PYTHON_VERSION"
conda create --name $ENV_NAME python=$PYTHON_VERSION -y

# Activate the newly created environment
echo "Activating the $ENV_NAME environment"
conda activate $ENV_NAME

# Verify if the correct conda environment is activated
echo
if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    echo "$CONDA_DEFAULT_ENV"
    echo Failed to activate conda environment.
    exit 1
else
    echo Successfully activated conda environment.
fi

# Install mamba release to boost installation and resolve dependencies
conda install -c conda-forge mamba=1.4.2 -y


# Install packages using conda or mamba
echo "Installing packages with conda or mamba"
conda install -c hussamalafandi cppprojection -c conda-forge -y

mamba install pytorch=2.2.1 torchvision=0.17.1 torchaudio=2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
mamba install conda-forge::wandb=0.16.3 -y
mamba install conda-forge::natsort=8.4.0 -y
mamba install conda-forge::tabulate=0.9.0 -y
mamba install conda-forge::conda-build=24.1.2 -y
mamba install conda-forge::mp_pytorch=0.1.4 -y
mamba install conda-forge::cw2=2.5.1 -y

# Add the current TCE repo to python path
conda develop .

# Download packages from Github

# # Fancy_Gym
# git clone -b tce_final --single-branch git@github.com:BruceGeLi/fancy_gymnasium.git
# cd fancy_gymnasium
# pip install -e .
# conda develop .
# cd ..

# Trust_Region_Projection
git clone -b TCE_ICLR24 --single-branch git@github.com:BruceGeLi/trust-region-layers.git
cd trust-region-layers
conda develop .
cd ..

# Git_Repo_Tracker
git clone -b main --single-branch git@github.com:ALRhub/Git_Repos_Tracker.git
cd Git_Repos_Tracker
pip install -e .
conda develop .
cd ..

# MetaWorld
git clone -b tce_final --single-branch git@github.com:BruceGeLi/Metaworld.git
cd Metaworld
pip install -e .
conda develop .
cd ..

# Install packages using pip
echo "Installing packages with pip"
pip install --upgrade pip
# pip install stable-baselines3==2.2.1
pip install warp-lang
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install isaacsim isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app --extra-index-url https://pypi.nvidia.com
cd ..
cd fancy_gym
pip install -e .
cd ..
cd stable-baselines3
pip install -e .
cd ..
cd pytorch_kinematics
pip install -e .
pip install arm_pytorch_utilities
cd ..
cd alr_tasks/exts/alr_isaaclab_tasks
pip install -e .
cd ../../../

#Install IsaacLab
cd IsaacLab
./isaaclab.sh -i
cd ..


echo "Configuration completed successfully."

