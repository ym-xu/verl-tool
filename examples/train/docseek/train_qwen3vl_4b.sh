set -x
# ========== DocSeek VQA-only: Qwen3-VL-4B on 4×L40S ==========

dataset_name=docseek/v2_vqa_only
train_data=[$(pwd)/data/${dataset_name}/train.parquet]
val_data=[$(pwd)/data/${dataset_name}/val.parquet]
model_name=Qwen/Qwen3-VL-4B-Instruct

# Image resolution: 512*28*28 = 401408 (~400K pixels)
max_pixels=401408
min_pixels=3136

# RL algorithm
rl_alg=grpo
n=8                    # GRPO group size (8 samples per prompt)
batch_size=48          # total batch = 6 unique prompts × 8 = 48 (divisible by 3 GPUs)
ppo_mini_batch_size=24

# GPU config (3×RTX PRO 6000 Blackwell 98GB)
n_gpus_per_node=3
n_nodes=1
tensor_model_parallel_size=1  # 4B fits on single 98GB easily
gpu_memory_utilization=0.85   # 98GB, plenty of room
do_offload=True               # offload to avoid CPU OOM from Ray workers
strategy="fsdp2"
fsdp_size=-1
ulysses_sequence_parallel_size=1

# Sequence lengths
max_prompt_length=16384       # image tokens can be large (some >8K)
max_response_length=8192
max_obs_length=8192
ppo_max_token_len_per_gpu=$(expr $max_prompt_length + $max_response_length)

# Agent / tool config
enable_agent=True
action_stop_tokens='</tool_call>'
max_turns=2                   # at most 1 zoom
mask_observations=True
enable_mtrl=True
max_action_length=2048

# Training
lr=1e-6
temperature=1.0
top_p=1.0
kl_loss_coef=0.0
kl_coef=0
entropy_coeff=0
kl_loss_type=low_var_kl

# Micro batch (per GPU)
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=8

# Reward
reward_manager=docseek

# Qwen3-VL special tokens
additional_eos_token_ids=[151645]  # <|im_end|>

# vLLM
use_dynamic_bsz=False
max_num_batched_tokens=5000
rollout_mode='async'

# Run name
model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name_postfix="-vqa-only"
run_name="${reward_manager}-${strategy}-agent-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-lr${lr}${run_name_postfix}"

export VERL_RUN_ID=$run_name
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES=1,2,3

# Action stop tokens temp file
action_stop_tokens_file="$(pwd)$(mktemp)"
mkdir -p $(dirname $action_stop_tokens_file)
echo -e -n "$action_stop_tokens" | tee $action_stop_tokens_file
echo "action_stop_tokens_file=$action_stop_tokens_file"

# Start DocSeek tool server
host=$(hostname -i | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.serve --host $host --port $port --tool_type "docseek" --workers_per_tool 8 &
server_pid=$!
echo "DocSeek tool server (pid=$server_pid) started at $tool_server_url"
sleep 10  # wait for server to start

PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.dataloader_num_workers=4 \
    data.train_batch_size=$batch_size \
    data.val_batch_size=100 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    +data.max_pixels=$max_pixels \
    +data.min_pixels=$min_pixels \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.agent.enable_agent=$enable_agent \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    actor_rollout_ref.agent.max_action_length=$max_action_length \
    actor_rollout_ref.agent.max_concurrent_trajectories=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.model.fsdp_config.fsdp_size=$fsdp_size \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console','wandb'] \
    trainer.project_name=docseek \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=200

pkill -P -9 $server_pid
kill -9 $server_pid
