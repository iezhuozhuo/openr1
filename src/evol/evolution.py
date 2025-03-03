
from loguru import logger
from src.evol.utils import clean_solution, remove_dup_solutions, read_prompt, format_input, softmax
from src.evol.fitness import cosine_scaled_reward, accuracy_reward, format_reward, cosine_lang_reward

class EvolOpt(object):

    def __init__(self, args, client, tokenizer):
        """进化问题problem的solution
            client: 部署的vllm的client
            pop_size: 初始种群大小，也就是初次生成的个数；
            tokenizer：分词器用于进行fitness计算
        """
        self.args = args
        self.client = client
        self.tokenizer = tokenizer

    def init_population(self, problem):

        gen_pmt = read_prompt(self.args.gen_responses_prompt)
        gen_pmt = gen_pmt.format_map({"problem": problem, "occup": "<your answer>"})
        gen_inputs = format_input(gen_pmt, self.tokenizer)
        # logger.info(f"Formated input:\n {gen_inputs}")

        cnt = 0
        stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
        solutions = []
        gen_responses_temp = self.args.gen_responses_temp
        while len(solutions) < self.args.pop_size:
            # 生成多样性回复困难则提高温度
            if cnt > 5:
                logger.debug(f"难度较大，需要提高温度")
                gen_responses_temp = gen_responses_temp + 0.1

            response = self.client.completions.create(
                model=self.args.model_path,  # 与启动时指定的模型名称一致
                prompt=gen_inputs,
                max_tokens=self.args.max_response_len,
                temperature=gen_responses_temp,
                top_p=0.95,
                # repetition_penalty=1.1,
                n=self.args.pop_size,
                stop=stop_words,
            )
            for chose in response.choices:
                solutions.append(clean_solution(chose.text))

            # 如果生成的多样性回复难度太大 那么不需要在去重了
            if cnt < 10:
                solutions = remove_dup_solutions(solutions)
                cnt += 1
            else:
                logger.debug(f"难度很大，丢弃去重操作")
                break

        return solutions

    def calculate_fitness(self, solutions, gt_ans):
        len_rwd = cosine_scaled_reward(gt_ans, solutions, self.tokenizer)
        print(f"cosin length rewards: {len_rwd}\n")

        acc_rwd = accuracy_reward(gt_ans, solutions)
        print(f"answer rewards: {acc_rwd}\n")

        format_rwd = format_reward(gt_ans, solutions)
        print(f"format reward: {format_rwd}\n")

        lang_rwd = cosine_lang_reward(gt_ans, solutions, self.tokenizer)
        print(f"lang reward: {lang_rwd}\n")

        rwd = []
        for i in range(len(acc_rwd)):
            rwd.append(float(len_rwd[i] + acc_rwd[i] + format_rwd[i] + lang_rwd[i]))
        print(f"final rewards: {rwd}\n")

        sel_p = softmax(rwd)
        print(f"softmax prob: {sel_p}")

