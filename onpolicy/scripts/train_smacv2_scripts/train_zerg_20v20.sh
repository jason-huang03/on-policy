env="StarCraft2v2"
map="10gen_zerg"
algo="rmappo"
units="20v20"

exp="tune2"
seed_max=1

GPU_FREE_MOST=$(./gpu_free.sh |  sort -t',' -k2 -n -r | head -n1)
GPU_FREE_MOST=${GPU_FREE_MOST//,/}
GPU_ID=$(echo ${GPU_FREE_MOST} | awk '{print $1}')
GPU_FREE_MEM=$(echo ${GPU_FREE_MOST} | awk '{print $2}')
CPU_USAGE=$(./cpu_usage.sh)

echo "cpu usage is ${CPU_USAGE}%, training will start in 10 seconds"

sleep 5


echo "we use gpu ${GPU_ID} with the most free memory. free mem is ${GPU_FREE_MEM} MB"

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=$GPU_ID python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --units ${units} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 \
    --num_env_steps 20000000 --ppo_epoch 5 --use_value_active_masks --use_eval --eval_episodes 32 \
    --gamma 0.99 --lambda 0.95 --huber_delta 10.0 --max_grad_norm 10.0 --data_chunk_length 10 --layer_N 2 
done

echo "training is going background"

