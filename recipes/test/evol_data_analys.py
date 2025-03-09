from transformers import AutoTokenizer

import pickle
import sys
import numpy as np
sys.path.append("/extrahome0/Zhuo/code/open-r1")
from src.evol.fitness import extract_answer
from src.evol.fitness import cosine_scaled_reward, accuracy_reward, format_reward, cosine_lang_reward


def calculate_rwd(solutions, gt_ans, tokenizer):
    len_rwd = cosine_scaled_reward(gt_ans, solutions, tokenizer)
    acc_rwd = accuracy_reward(gt_ans, solutions)
    format_rwd = format_reward(gt_ans, solutions)
    lang_rwd = cosine_lang_reward(gt_ans, solutions, tokenizer)
    return len_rwd, acc_rwd, format_rwd, lang_rwd


def calculate_fitness(solutions, gt_ans, tokenizer):
    len_rwd, acc_rwd, format_rwd, lang_rwd = calculate_rwd(solutions, gt_ans, tokenizer)

    rwd = []
    details_info = []
    for i in range(len(acc_rwd)):
        rwd.append(float(len_rwd[i] + acc_rwd[i] + format_rwd[i] + lang_rwd[i]))
        details_info.append(
            {"rwd": rwd[i], "len_rwd": len_rwd[i], "acc_rwd": acc_rwd[i],
             "format_rwd": format_rwd[i], "lang_rwd": lang_rwd[i]}
        )
    return rwd, details_info


tokenizer = AutoTokenizer.from_pretrained("/extrahome0/HF_models/Qwen2.5-7B-Instruct")

data_path = "/extrahome0/Zhuo/output/evol_s1k/v1/evol_s1k.pkl"
# data_path = "/extrahome0/Zhuo/output/evol_s1k/v1/evol_s1k_-.pkl"
with open(data_path, "rb") as file:
    dps = pickle.load(file)
for dp in dps:
    all_solutions = dp["all_solutions"]
    new_all_solutions = []
    for solution in all_solutions:
        if len(solution.split("#")) > 1:
            # 删除多余的阐述语句
            solution = "#".join(solution.split("#")[1:-1]).replace("Refined", "")
        new_all_solutions.append(solution)
    dp["all_solutions"] = new_all_solutions

w2r, w2w, r2w = 0, 0, 0
init_r_ratio = []
evol_r_ratio = []

real_evol = 0
can_evol = 0
evol_fitness = []
init_fit = []
evol_fit = []

sel_evol = 0

for dp in dps:
    all_solutions = dp["all_solutions"]
    init_solutions = all_solutions[0:-6]
    evol_solutions = all_solutions[-6:]
    gt_answer = extract_answer(dp['attempt'])

    init_rwd, init_details = calculate_fitness(init_solutions, gt_answer, tokenizer)
    init_right = False
    init_r_cnt = 0
    for init_info in init_details:
        if init_info["acc_rwd"] >= 0.5:
            init_r_cnt += 1
            init_right = True

    evol_rwd, evol_details = calculate_fitness(evol_solutions, gt_answer, tokenizer)
    evol_right = False
    evol_r_cnt = 0
    for evol_info in evol_details:
        if evol_info["acc_rwd"] >= 0.5:
            evol_r_cnt += 1
            evol_right = True

    # 进化有多少的样本从不正确到正确
    if not init_right and evol_right:
        w2r += 1
    if init_right and not evol_right:
        r2w += 1
    if not init_right and not evol_right:
        w2w += 1

    # 进化前后的正确率算不算w2w
    # if not init_right and not evol_right:
    #     continue  # 没有进化出来正确答案
    # 进化前后的正确率
    init_r_ratio.append(init_r_cnt / len(init_solutions))
    evol_r_ratio.append(evol_r_cnt / len(evol_solutions))

    # 进化有多少的样本的fitness提升了
    init_fitness_vals = np.array(init_rwd)
    evol_fitness_vals = np.array(evol_rwd)
    # 找到当前代的最优解
    max_fitness_idx = np.argmax(init_fitness_vals)
    min_fitness_idx = np.argmin(evol_fitness_vals)

    if init_fitness_vals[max_fitness_idx] < evol_fitness_vals[min_fitness_idx]:
        real_evol += 1
    if np.mean(evol_rwd) > np.mean(init_rwd):
        can_evol += 1
    # evol_fitness.append(evol_fitness_vals[min_fitness_idx]-init_fitness_vals[max_fitness_idx])
    evol_fitness.append(np.mean(evol_rwd)-np.mean(init_rwd))
    init_fit.append(np.mean(init_rwd))
    evol_fit.append(np.mean(evol_rwd))

    # 多少最优样本选择进化的
    fitness_vals = np.array(init_rwd + evol_rwd)
    max_fit_idx = np.argmax(fitness_vals)
    if max_fit_idx >= len(all_solutions) - 6:
        sel_evol += 1

    # TODO 进化之后多样性变化


print(f"总数据量: {len(dps)}")
print(f"将错误进化到正确: {w2r/len(dps):.3f}, "
      f"将正确进化到错误: {r2w/len(dps):.3f}, "
      f"无法进化出正确的: {w2w/len(dps):.3f}")
print(f"原始正确率: {np.mean(init_r_ratio):.3f}, 正确率中位数: {np.median(init_r_ratio):.3f}\n"
      f"进化正确率: {np.mean(evol_r_ratio):.3f}, 正确率中位数: {np.median(evol_r_ratio):.3f}")
print(f"原始fitness: {np.mean(init_fit):.3f}, 进化fitness: {np.mean(evol_fit):.3f}\n"
      f"完全提升fitness率: {real_evol}/{len(dps)}, 能够提升fitness率: {can_evol/len(dps):.3f}, "
      f"平均提升fitness: {np.mean(evol_fitness):.3f}")
print(f"选择进化的回复率: {sel_evol/len(dps):.3f}")