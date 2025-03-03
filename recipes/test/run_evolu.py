
import sys
import yaml
from openai import OpenAI
from typing import Optional
from loguru import logger
from dataclasses import dataclass, field


from transformers import HfArgumentParser
from transformers import AutoTokenizer

sys.path.append("/extrahome0/Zhuo/code/open-r1")
from src.evol.evolution import EvolOpt



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

    # test_problem
    problem = ("A square is inscribed in a right triangle with legs of lengths 6 and 8, such that the square shares "
               "the right angle with the triangle. Find the side length of the square.")
    gt_ans = '\\frac{24}{7}'
    solutions = eval_opt.init_population(problem)
    print(eval_opt.calculate_fitness(solutions, gt_ans))


if __name__ == "__main__":
    main()
