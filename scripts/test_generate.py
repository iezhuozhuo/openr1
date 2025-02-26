
import os
import sys
import json
import yaml
from vllm import LLM, SamplingParams

sys.path.append("/extrahome0/Zhuo/code/open-r1")
from src.open_r1.rewards import accuracy_reward, format_reward

file_path = sys.argv[1]
with open(file_path, "r") as file:
    config = yaml.safe_load(file)
model_path = config["model_path"]
available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
pipeline_parallel_size = 1
# model_name_or_path = "/extrahome0/HF_models/Qwen2.5-32B-Instruct"  # Exchange with another smol distilled r1

llm = LLM(
    model=model_path,
    tensor_parallel_size=len(available_gpus) // pipeline_parallel_size,
    pipeline_parallel_size=pipeline_parallel_size,
    trust_remote_code=True,
)

while True:
    file_path = input("请输入测试输入（或输入 'exit' 退出）：").strip()
    if file_path.lower() == 'exit':
        print("退出程序。")
        break

    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    # config = OmegaConf.load(file_path)
    problem = config["problem"]
    answer = config["answer"]
    system_prompt = config["system_prompt"]
    generated_num = config["n"]

    input_dict = {"system_prompt": system_prompt, "problem": problem}
    inputs = "{system_prompt}\nStudent: {problem}\nYou: ".format_map(input_dict)

    outputs = llm.generate(
        inputs,
        SamplingParams(
            temperature=config["temperature"],
            top_p=0.95,
            max_tokens=config["max_tokens"],
            n=generated_num,
        ),
    )
    outputs = sorted(
        outputs, key=lambda x: int(x.request_id)
    )  # sort outputs by request_id

    # 单测评
    output_file = os.path.join(config["output_path"], f"{config['name']}.jsonl")
    with open(output_file, "w", encoding='utf-8') as f:
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = []
            for idx in range(len(output.outputs)):
                opt = output.outputs[idx].text
                acc_rw = accuracy_reward([[{"content": opt}]], [answer])[0]
                fmt_rw = format_reward([[{"content": opt}]])[0]
                generated_text.append(opt)
                json.dump({"problem": problem, f"response {idx}": opt, "acc_reward": acc_rw, "format_reward": fmt_rw},
                          f, ensure_ascii=False)
                f.write("\n")
            # print(generated_text)
