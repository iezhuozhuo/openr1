
#accelerate launch --config_file=recipes/accelerate_configs/ddp.yaml src/open_r1/sft.py \
#--model_name_or_path /extrahome0/HF_models/Qwen2.5-Math-1.5B-Instruct \
#--dataset_name /extrahome0/HF_datasets/Bespoke-Stratos-17k \
#--learning_rate 2.0e-5 \
#--num_train_epochs 1 \
#--packing \
#--max_seq_length 4096 \
#--per_device_train_batch_size 2 \
#--gradient_accumulation_steps 16 \
#--gradient_checkpointing \
#--bf16 \
#--logging_steps 5 \
#--eval_strategy steps \
#--eval_steps 100 \
#--output_dir /extrahome0/Zhuo/output/openr1/Qwen2.5-1.5B-Ist-Distill


# Train via YAML config
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml