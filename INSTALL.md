# Installing Conda Environment from Zero to Hero

The following guidance works well for a machine with 3090/A40/A800/A100 GPU, cuda 12.1, driver 515.65.01.

First, git clone this repo and `cd` into it.

```
git clone https://github.com/KolaKivy/AFRO.git
cd AFRO
```

---

1.create python/pytorch env

    conda remove -n afro --all
    conda create -n afro python=3.8
    conda activate afro


---

2.install torch

    # if using cuda>=12.1
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    # else, 
    # just install the torch version that matches your cuda version

---

3.install huggingface_hub, pyyaml and safetensors

```
pip install huggingface_hub==0.20.3 pyyaml safetensors
```

---

4.install afro workspace

    pip install -e .


---

5.install mujoco in `~/.mujoco`

    cd
    mkdir .mujoco
    cd .mujoco
    wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz --no-check-certificate
    
    tar -xvzf mujoco210.tar.gz

and put the following into your bash script (usually in `YOUR_HOME_PATH/.bashrc`). Remember to `source ~/.bashrc` to make it work and then open a new terminal.

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
    export MUJOCO_GL=egl

Before installing mujoco-py, you must first install the following development packages:

```
# if using Ubuntu / Debian
sudo apt update
sudo apt install -y libx11-dev libxext-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl1-mesa-dev libglu1-mesa-dev libegl1-mesa-dev libglew-dev libvulkan1
# if using CentOS / Rocky
sudo yum install -y libX11-devel libXext-devel libXrandr-devel libXinerama-devel libXcursor-devel libXi-devel mesa-libGL-devel mesa-libGLU-devel mesa-libEGL-devel glew-devel vulkan-loader
```

and then install mujoco-py (in the folder of `third_party`):

    cd YOUR_PROJECT_PATH
    cd third_party/mujoco-py-2.1.2.14
    pip install -e .
    cd ..


----

6.install sim env (in the folder of `third_party`):

    pip install setuptools==80.10.2 Cython==0.29.35 patchelf==0.17.2.0

    cd gym-0.21.0 && pip install -e . && cd ..
    cd Metaworld && pip install -e . && cd ..
    cd rrl-dependencies && pip install -e mj_envs/. && pip install -e mjrl/. && cd ..

download Adroit RL experts from [OneDrive](https://1drv.ms/u/s!Ag5QsBIFtRnTlFWqYWtS2wMMPKNX?e=dw8hsS), unzip it, and put the `ckpts` folder under `$YOUR_REPO_PATH/third_party/VRL3/`.

---

7.install pytorch3d (a simplified version)

    cd pytorch3d_simplified && pip install --no-build-isolation -e . && cd ..


---

8.install some necessary packages

    pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor natsort timm


---

