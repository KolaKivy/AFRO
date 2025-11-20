# Examples:
# bash scripts/train_policy.sh dp3 adroit_hammer 0322 0 0
# bash scripts/train_policy.sh dp3 dexart_laptop 0322 0 0
# bash scripts/train_policy.sh simple_dp3 adroit_hammer 0322 0 0
# bash scripts/train_policy.sh dp3 metaworld_basketball 0602 0 0



DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
data_type=${5}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data_${data_type}/outputs/${exp_name}_seed${seed}"
zarr_path="data_${data_type}/${task_name}_expert.zarr"       # 这里记得改
eval_episodes=50
max_train_episodes=99999


# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=${6}
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

cd AFRO


export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}
python train.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            task.dataset.zarr_path=${zarr_path} \
                            task.env_runner.eval_episodes=${eval_episodes} \
                            task.dataset.max_train_episodes=${max_train_episodes}



                                