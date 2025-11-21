# Examples:
# bash scripts/train_afro_policy.sh afro_policy adroit_pen 0001 0 0

DEBUG=False
save_ckpt=True
pretrain_name="afro"

alg_name=${1}
task_name=${2}
config_name=${alg_name} 
addition_info=${3}
seed=${4}
pretrain_exp_name=${task_name}-${pretrain_name}-${addition_info}
exp_name=${task_name}-${alg_name}-${addition_info}
checkpoint="data/outputs/${pretrain_exp_name}_seed${seed}/checkpoints/pretrain_latest.ckpt"

run_dir="data/outputs_policy/${exp_name}_seed${seed}" 
zarr_path="data/${task_name}_expert.zarr"  


# gpu_id=$(bash scripts/find_gpu.sh)
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


checkpoint_every=10
max_train_episodes=1000000
resume=False
num_epochs=101
eval_episodes=50


echo -e "\033[32mcheckpoint_every: ${checkpoint_every}\033[0m"
echo -e "\033[32mresume: ${resume}\033[0m"
echo -e "\033[32mnum_epochs: ${num_epochs}\033[0m"
echo -e "\033[32mcheckpoint_path: ${checkpoint}\033[0m"


export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}
python train.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            task_whole_name=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            training.checkpoint_every=${checkpoint_every} \
                            task.env_runner.eval_episodes=${eval_episodes} \
                            training.resume=${resume} \
                            policy.checkpoint=${checkpoint} \
                            training.num_epochs=${num_epochs} \
                            task.dataset.zarr_path=${zarr_path} \
                            task.dataset.max_train_episodes=${max_train_episodes} \



                                