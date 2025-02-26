#

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
--num_processes=3 src/open_r1/grpo.py \
--config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml