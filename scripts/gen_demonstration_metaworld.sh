# bash scripts/gen_demonstration_metaworld.sh basketball
# conda activate ar3d
cd third_party/Metaworld

task_name=${1}

export CUDA_VISIBLE_DEVICES=3
python gen_demonstration_expert.py --env_name=${task_name} \
            --num_episodes 25 \
            --root_dir "../../afro_workspace/data/" \


# stick-pull        stick-push         pick-place-wall   sweep
# pick-place        pick-out-of-hole   push
# push-wall         peg-insert-side    coffee-pull       bin-picking
# peg-unplug-side   dial-turn          handle-press      soccer