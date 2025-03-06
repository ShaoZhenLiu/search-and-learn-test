"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from typing import List, Union

from datasets import Dataset

from sal.config import Config
from sal.utils.rewards.global_conf import THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END  # , OAI_RM_MODEL
from sal.utils.rewards.reward_types import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from sal.utils.rewards.math_utils import extract_answer, grade_answer_sympy, grade_answer_mathd


# 这里尝试使用orm进行评价
# from deepscaler.system_prompts import ORM_PROMPT
# from deepscaler.utils import call_gemini_llm, call_oai_rm_llm

# ORM_USER_TEMPLATE = """
# Problem: {problem}
# Answer 1: {answer_1}
# Answer 2: {answer_2}
# """

class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)

        problem = input.problem
        model_response = input.model_response

        # Extract solution.
        if self.config.check_think_format:
            if THOUGHT_DELIMITER_START in model_response and THOUGHT_DELIMITER_END in model_response:
                model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
            else:
                return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
        else:
            model_solution = model_response

        model_answer = extract_answer(model_solution)
        if model_answer is None:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # Process the ground truth(s)
        ground_truths = input.ground_truth.get("answer", None)
        if ground_truths is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]

        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)

        if not processed_ground_truths:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer,
                                                                                              ground_truth)
            if is_correct:
                return RewardOutput(reward=self.config.correct_reward, is_correct=True)

        # # If latex heuristics fail and ORM is enabled, use LLM as ORM to evaluate correctness
        # if self.config.use_math_orm:
        #     for ground_truth in processed_ground_truths:
        #         try:
        #             orm_response = call_gemini_llm(
        #                 system_prompt=ORM_PROMPT,
        #                 prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
        #                 temperature=0.0,
        #             )

        #             if "[[YES]]" in orm_response:
        #                 return RewardOutput(reward=self.config.correct_reward, is_correct=True)
        #         except Exception as e:
        #             print ("Error calling Gemini ORM, trying OAI RM")
        #             orm_response = call_oai_rm_llm(
        #                 system_prompt=ORM_PROMPT,
        #                 prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
        #                 temperature=0.0,
        #                 model_id=OAI_RM_MODEL,
        #             )

        #             if "[[YES]]" in orm_response:
        #                 return RewardOutput(reward=self.config.correct_reward, is_correct=True)
        #             continue

        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)


def _sal_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], enable_llm: bool = False,
                   check_think: bool = False):
    """reward_function: implimented both rule-based and orm to deal woth LLM math response checking

    Args:
        solution_str (str): math response generated by LLM
        ground_truth (Union[str, List[str]]): ground truth answer provided by the dataset
        enable_llm (bool, optional): whether to use orm when rule-base checking is fail. Defaults to False.
        check_think (bool, optional): whether to check think format, like <think> and </think> labels. Defaults to False.

    Returns:
        bool: whether LLM's answer is correct. True for correct, False for incorrect
    """
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    reward_config.check_think_format = check_think
    reward_fn = RewardMathFn(reward_config)
    reward_response = reward_fn(
        RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str,
                    ground_truth={"answer": ground_truth}))
    return reward_response.is_correct


def sal_reward_fn(dataset: Dataset, config: Config):
    assert config.approach in ["iter_gen_multi_turn", "diff_of_n_multi_turn"], \
        NotImplementedError("当前的方法还不能判断生成答案的正确与否，需要进一步验证")
    dataset = dataset.map(
        lambda x: {
            "correct": _sal_reward_fn(
                solution_str=x["messages"][-1]["content"],  # 最后一个回答会输出在\boxed{}中的答案
                ground_truth=x["answer"],
                enable_llm=False, check_think=False,
            )
        }
    )

    if config.filter_correct:
        total = len(dataset)
        dataset = dataset.filter(lambda x: x["correct"])  # 只保留正确的数据
        correct_count = len(dataset)
        acc = correct_count / total * 100
        return dataset, acc

    return dataset, None

# if __name__ == "__main__":
#     reward = RewardMathFn(RewardConfig)
#     # input = RewardInput(problem="Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$.",
#     #                     problem_type=RewardType.MATH,
#     #                     model_response="<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.",
#     #                     ground_truth={"answer": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"]})
#     input = RewardInput(problem="...",
#                         problem_type=RewardType.MATH,
#                         model_response="<think> I am omniscient. </think> The answer is \[\n\\boxed{\\dfrac{5}{3}}\n\]\n",
#                         ground_truth={"answer": ["\\frac{5}{3}"]})
#     output = reward(input)
#     print(output)
