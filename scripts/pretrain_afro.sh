# Examples:
# bash scripts/pretrain_afro.sh afro adroit_pen afro_ckpt 0 0

DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name=${alg_name} 
addition_info=${3}
seed=${4}
datanum=${6}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"  
zarr_path="data/${task_name}_expert.zarr"  

gpu_id=${5}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=offline
    echo -e "\033[33mTrain mode\033[0m"
fi

echo -e "\033[32mrun_dir: ${run_dir}\033[0m"
echo -e "\033[32mzarr_path: ${zarr_path}\033[0m"

cd 3D-Diffusion-Policy


export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

checkpoint_every=10
eval_episodes=50
max_train_episodes=1000000
resume=false

echo -e "\033[32mcheckpoint_every: ${checkpoint_every}\033[0m"
echo -e "\033[32mmax_train_episodes: ${max_train_episodes}\033[0m"
echo -e "\033[32mresume: ${resume}\033[0m"

python AFRO_Pretrain.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            training.checkpoint_every=${checkpoint_every} \
                            training.resume=${resume} \
                            task.dataset.zarr_path=${zarr_path} \
                            task.dataset.max_train_episodes=${max_train_episodes}
                            # task.env_runner.eval_episodes=${eval_episodes}



                                