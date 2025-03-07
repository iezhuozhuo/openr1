import os.path
import sys
import yaml
import pickle
from openai import OpenAI
from typing import Optional
from loguru import logger
from datasets import load_dataset
from dataclasses import dataclass, field

from transformers import HfArgumentParser
from transformers import AutoTokenizer

sys.path.append("/extrahome0/Zhuo/code/open-r1")
from src.evol.evolution import EvolOpt
from src.evol.fitness import extract_answer


@dataclass
class EvolutionArguments:
    config_path: Optional[str] = field(default=None, metadata={"help": "The configuration file to use."})


def main():
    # 读取配置
    logger.info("loading configuration")
    parser = HfArgumentParser(EvolutionArguments)
    args = parser.parse_args_into_dataclasses()[0]
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)
    for k, v in config.items():
        setattr(args, k, v)

    # 加载分词器
    logger.info("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 加载客户端
    logger.info("building client")
    client = OpenAI(
        base_url=args.base_url,  # vLLM服务的地址 端口1/2 8000/8012
        api_key=args.api_key,  # 如果设置了API密钥需要填写,
        # repetition_penalty=1.1,
    )

    # 定义Evolution Opt
    logger.info("loading Evolution Operations")
    eval_opt = EvolOpt(
        args=args, client=client, tokenizer=tokenizer
    )

    # 加载预测数据
    output_file = os.path.join(args.output_path, "evol_s1k.pkl")
    if os.path.isfile(output_file):
        with open(output_file, "rb") as file:
            dps = pickle.load(file)
            reuse_idx = len(dps)
    else:
        dps = []
        reuse_idx = 0
    logger.debug(f"Reuse {reuse_idx} data in {output_file}")

    train_data = load_dataset(args.problem_path)["train"]
    for idx in range(len(train_data)):
        if idx < reuse_idx:
            continue

        logger.debug(f"处理第 {idx} 个问题")
        dp = train_data[idx]
        problem = dp["question"]
        gt_ans = extract_answer(dp["attempt"])
        if not gt_ans:
            ds = eval(dp["metadata"])
            if "answer" in ds:
                gt_ans = ds['answer']
            else:
                continue

        eval_solution, all_solutions = eval_opt.pipeline(problem, gt_ans)
        dp["evol_solution"] = eval_solution
        dp["all_solutions"] = all_solutions
        dp["idx"] = idx
        dps.append(dp)

        if idx > 0 and idx % 5 == 0:
            with open(output_file, "wb") as file:
                pickle.dump(dps, file)


    # test_problem
    # problem = ("A square is inscribed in a right triangle with legs of lengths 6 and 8, such that the square shares "
    #            "the right angle with the triangle. Find the side length of the square.")
    # gt_ans = '\\frac{24}{7}'
    # eval_opt.pipeline(problem, gt_ans)

    # solutions = eval_opt.init_population(problem)
    # solution_1 = "To find the side length of the square inscribed in a right triangle with legs of lengths 6 and 8, " \
    #              "we start by visualizing the problem. Let's denote the right triangle as \\(\\triangle ABC\\) with " \
    #              "\\( \\angle C = 90^\\circ \\), \\(AC = 6\\), and \\(BC = 8\\). The square is inscribed such that " \
    #              "one of its vertices is at \\(C\\) and the other two vertices lie on \\(AC\\) and \\(BC\\). Let the " \
    #              "side length of the square be \\(s\\).\n\nWe can place the square such that one of its sides lies " \
    #              "along the legs of the triangle. This means that the square cuts off two smaller right triangles " \
    #              "from the original triangle. The legs of these smaller right triangles are \\(6-s\\) and \\(8-s\\), " \
    #              "and the hypotenuse is the same as the original hypotenuse minus the side of the square.\n\nThe area " \
    #              "of the original triangle can be calculated as:\n\\[\n\\text{Area} = \\frac{1}{2} \\times 6 \\times " \
    #              "8 = 24.\n\\]\n\nThe area can also be expressed as the sum of the area of the square and the areas " \
    #              "of the two smaller right triangles:\n\\[\n\\text{Area} = s^2 + \\frac{1}{2} \\times s \\times (6-s) " \
    #              "+ \\frac{1}{2} \\times s \\times (8-s).\n\\]\n\nSimplifying the right-hand side:\n\\[\n24 = s^2 + " \
    #              "\\frac{1}{2} s (6 + 8 - s) = s^2 + \\frac{1}{2} s (14 - s) = s^2 + 7s - \\frac{1}{2} s^2 = \\frac{" \
    #              "1}{2} s^2 + 7s.\n\\]\n\nMultiplying through by 2 to clear the fraction:\n\\[\n48 = s^2 + " \
    #              "14s.\n\\]\n\nRearranging the equation into standard quadratic form:\n\\[\ns^2 + 14s - 48 = " \
    #              "0.\n\\]\n\nWe solve this quadratic equation using the quadratic formula \\(s = \\frac{-b \\pm " \
    #              "\\sqrt{b^2 - 4ac}}{2a}\\), where \\(a = 1\\), \\(b = 14\\), and \\(c = -48\\):\n\\[\ns = \\frac{-14 " \
    #              "\\pm \\sqrt{14^2 - 4 \\cdot 1 \\cdot (-48)}}{2 \\cdot 1} = \\frac{-14 \\pm \\sqrt{196 + 192}}{2} = " \
    #              "\\frac{-14 \\pm \\sqrt{388}}{2} = \\frac{-14 \\pm 2\\sqrt{97}}{2} = -7 \\pm \\sqrt{" \
    #              "97}.\n\\]\n\nSince \\(s\\) must be a positive length, we take the positive root:\n\\[\ns = -7 + " \
    #              "\\sqrt{97}.\n\\]\n\nTo verify, we can use the geometric relationship. The side length \\(s\\) can " \
    #              "also be found by considering the similar triangles formed. The ratio of the sides of the smaller " \
    #              "triangles to the original triangle is the same as the ratio of the side of the square to the leg of " \
    #              "the triangle minus the side of the square. This gives us:\n\\[\n\\frac{s}{6-s} = \\frac{8-s}{" \
    #              "s}.\n\\]\n\nCross-multiplying gives:\n\\[\ns^2 = (6-s)(8-s) = 48 - 14s + s^2.\n\\]\n\nSimplifying, " \
    #              "we get:\n\\[\n0 = 48 - 14s \\implies s = \\frac{48}{14} = \\frac{24}{7}.\n\\]\n\nThus, " \
    #              "the side length of the square is:\n\\[\n\\boxed{\\frac{24}{7}}.\n\\] "
    # solution_2 = 'To find the side length of the square inscribed in a right triangle with legs of lengths 6 and 8, ' \
    #              'we start by visualizing the problem. Let the right triangle have vertices at \\( (0,0) \\), \\( (6,' \
    #              '0) \\), and \\( (0,8) \\). The square will have one vertex at the right angle of the triangle, ' \
    #              'and its sides will be parallel to the legs of the triangle.\n\nLet the side length of the square be ' \
    #              '\\( s \\). The square will touch the hypotenuse of the triangle. The coordinates of the vertices of ' \
    #              'the square can be described as follows:\n- One vertex at \\( (0,0) \\)\n- Another vertex at \\( (s,' \
    #              '0) \\)\n- Another vertex at \\( (0,s) \\)\n- The fourth vertex at \\( (s,s) \\)\n\nThe equation of ' \
    #              'the hypotenuse of the triangle can be found using the two points \\( (6,0) \\) and \\( (0,' \
    #              '8) \\). The slope of the hypotenuse is:\n\\[\n\\text{slope} = \\frac{8-0}{0-6} = -\\frac{4}{' \
    #              '3}\n\\]\nThe equation of the line in slope-intercept form is:\n\\[\ny = -\\frac{4}{3}x + ' \
    #              '8\n\\]\nSince the fourth vertex of the square \\( (s,s) \\) lies on this line, we substitute \\( x ' \
    #              '= s \\) and \\( y = s \\) into the equation:\n\\[\ns = -\\frac{4}{3}s + 8\n\\]\nTo solve for \\( s ' \
    #              '\\), we first eliminate the fraction by multiplying every term by 3:\n\\[\n3s = -4s + ' \
    #              '24\n\\]\nAdding \\( 4s \\) to both sides gives:\n\\[\n7s = 24\n\\]\nDividing both sides by 7, ' \
    #              'we get:\n\\[\ns = \\frac{24}{7}\n\\]\nThus, the side length of the square is \\(\\boxed{\\frac{24}{' \
    #              '7}}\\). '
    # new_solution = eval_opt.crossover(problem, solution_1, solution_2, gt_ans)
    # mutation = eval_opt.mutation(problem, new_solution, gt_ans)


if __name__ == "__main__":
    main()
