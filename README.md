**[Project Page]** | **arXiv**(soon to come)

**[Qiwei Liang](https://kolakivy.github.io/)**, Boyang Cai, Minghao Lai, Sitong Zhuang, Tao Lin, Yan Qin, Yixuan Ye, Jiaming Liang, [Renjing Xu](https://openreview.net/profile?id=~Renjing_Xu1)

<p align="center">
<img src="./teaser.pdf" width="80%"/>
</p>
# 💻 Installation
## Set up the environment
1. create python/pytorch env
```bash
conda create -n afro python=3.8 # isaacgym requires python <=3.8
conda activate afro

git clone https://github.com/KolaKivy/AFRO.git
```
2. Then follow the step 2 to 6 referred to the [installation of DP3]([3D-Diffusion-Policy/INSTALL.md at master · YanjieZe/3D-Diffusion-Policy](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/master/INSTALL.md))
3. install some necessary packages
```
pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor 
```

# 🛠️ Usage
1. **Generate demonstrations in simulation**: 
    ```shell
    bash scripts/gen_demonstration_adroit.sh hammer
    bash scripts/gen_demonstration_metaworld.sh bin-picking
    ```
    
    The first command will generate task `hammer` in Adroit, the second command line will generate task `bin-picking` in Metaworld. You can refer to DP3 to see all kinds of task supported in both Adroit and Metaworld. Tasks in dexart are also suppoted but wo don't test them in our work. **Note**: we default to collect 100 pairs of demonstration in adroit env and 25 in metaworld env, you can easily change this just in `gen_demonstration_*.sh`

2. Train AFRO: 3D self-supervised pipeline to pretrain the visual encoder
	```shell
	bash scripts/pretrain_afro.sh afro adroit_door afro_pretrain_data100 0 0
	```
	arg2: config_name   (here refers to afro.yaml in config)
	arg3: task_name
	arg4: additional information to name the outputs directory
	arg5: seed
	arg6: GPU_id

3. Train state_encoder and diffusion head:
	```shell
	bash scripts/train_afro_policy.sh afro_policy adroit_door afro_policy_data100 0 0
	```
	**Note**: You should open `scripts/train_afro_policy.sh` to modify your pretrained_visual_encoder checkpoint_path to make sure importing right weights

4. (Optional)Train state_encoder and diffusion head, Meanwhile fine-tune pre-trained visual_encoder weights
	```shell
	bash scripts/train_afro_policy_ft.sh afro_policy adroit_door afro_policy_ft_data100 0 0
	```
	This will train state_encoder and diffusion head with learing rate 1.0e-4 and fine-tune visual_encoder weights with learing rate 1.0e-5


# 🤖 Real Robot
**Hardware Setup**
1. Franka Robot
2. **L515** Realsense Camera (**Note: using the RealSense D435 camera might lead to failure of DP3 due to the very low quality of point clouds**)

**Software**
1. Ubuntu 20.04.01 (tested)
2. [Franka Interface Control](https://frankaemika.github.io/docs/index.html)
3. [Frankx](https://github.com/pantor/frankx) (High-Level Motion Library for the Franka Emika Robot)

**Data collect**
We manually collect expert demonstrations by remote operation and zip the raw data into `zarr` format aligning to simulation data. Remember to create a new yaml in `config/task`  folder under the name of your_realworld_task

**Real-world AFRO pretrain**
Different from pre-trainging in simulation，we replace pointnet encoder in simulation with pointtransformer encoder  in real-world experiment. You can pre-train with pointtransformer encoder by modifying `pointnet_type` in `afro.yaml`, then use the same command line as simulation:
```shell
	bash scripts/pretrain_afro.sh afro your_realworld_task afro_pretrain_pointtransformer 0 0
```
By the way, you ought to change the `pointnet_type` in `afro_policy.yaml` when you train the whole policy in real-world experiment

# 📝 Citation

If you find our work useful, please consider citing:

```
@article{liang2025whole,
  title={Whole-Body Coordination for Dynamic Object Grasping with Legged Manipulators},
  author={Liang, Qiwei and Cai, Boyang and He, Rongyi and Li, Hui and Teng, Tao and Duan, Haihan and Huang, Changxin and Zeng, Runhao},
  journal={arXiv preprint arXiv:2508.08328},
  year={2025}
}```
