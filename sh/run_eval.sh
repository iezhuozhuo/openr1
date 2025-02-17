
NUM_GPUS=4
#MODEL=/extrahome0/HF_models/DeepSeek-R1-Distill-Qwen-1.5B
#MODEL=/extrahome0/HF_models/Qwen2.5-Math-1.5B-Instruct
#MODEL=/extrahome0/HF_models/Qwen2.5-1.5B-Instruct
MODEL=/extrahome0/Zhuo/output/openr1/Qwen2.5-1.5B-Ist-Distill
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.9"
OUTPUT_DIR=data/evals/$MODEL

# MATH-500
TASK=math_500
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

## GPQA Diamond
#TASK=gpqa:diamond
#lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#    --custom-tasks src/open_r1/evaluate.py \
#    --use-chat-template \
#    --output-dir $OUTPUT_DIR
#
## AIME 2024
#TASK=aime24
#lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#    --custom-tasks src/open_r1/evaluate.py \
#    --use-chat-template \
#    --output-dir $OUTPUT_DIR


# multiple GPUs
#NUM_GPUS=8
#MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
#MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.8"
#TASK=aime24
#OUTPUT_DIR=data/evals/$MODEL
#
#lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#    --custom-tasks src/open_r1/evaluate.py \
#    --use-chat-template \
#    --output-dir $OUTPUT_DIR

#  large models which require sharding across GPUs
#NUM_GPUS=8
#MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
#MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.8"
#TASK=aime24
#OUTPUT_DIR=data/evals/$MODEL
#
#export VLLM_WORKER_MULTIPROC_METHOD=spawn
#lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#    --custom-tasks src/open_r1/evaluate.py \
#    --use-chat-template \
#    --output-dir $OUTPUT_DIR