from jinja2 import Template

step_prompt = {
    "step1": "Generate a new problem-solving approach:"
                 "\nPlease propose a new problem-solving approach that is significantly different from the previous solution."
                 "\nThe previous solution is: [{{ approach_1 }}]."
                 "\n\nProvide a detailed explanation of the logic and steps of the new approach, ensuring it does not rely on the previous method."
                 "\nThe problem is: [{{ problem }}].",
    "step2": "Compare the results of two problem-solving approaches:"
               "\nPlease compare two approaches for the following problem: {{ problem }}"
               "\n- Approach 1: {{ approach_1 }}"
               "\n- Approach 2: {{ approach_2 }}"
               "\n\nAnalyze whether the results of these two methods are consistent and provide your reasoning.",
    "step3": "Determine which result is more reliable:"
                 "\nIf the results of Approach 1 and Approach 2 are inconsistent, please analyze which result is more reliable based on the following criteria: "
                 "\n- Is the logic of the approach rigorous?"
                 "\n- Is the diff_of_n_data used complete and reliable?"
                 "\n- Does the method align with practical applications?"
                 "\n\n- Problem description: {{ problem }}"
                 "\n- Approach 1: {{ approach_1 }}"
                 "\n- Approach 2: {{ approach_2 }}"
                 "\n- Comparison analysis: {{ comparison }}"
                 "\n\nRefer to the above information to provide a step-by-step analysis and your conclusion.",
}

data = {
    "problem": "Compute: $1-2+3-4+5- \\dots +99-100$.",
    "solution": "$(1-2)+(3-4)+ \\dots +(97-98)+(99-100) = 50(-1) = \\boxed{-50}.$", "answer": "-50",
    "subject": "Intermediate Algebra", "level": 1, "unique_id": "test\/intermediate_algebra\/134.json",
    "step0_res": "<|start_header_id|>assistant<|end_header_id|>"
                 "\n\n## Step 1: Pair the numbers in the sequence"
                 "\nWe notice that we can pair the numbers in the sequence: $(1-2) + (3-4) + (5-6) + \\dots + (99-100)$. Each pair will result in a negative number since we are subtracting a larger number from a smaller one."
                 "\n\n## Step 2: Calculate the sum of the first 50 negative numbers"
                 "\nSince we have 50 pairs, we can calculate the sum of the first 50 negative numbers: $- (2 + 4 + 6 + \\dots + 100)$."
                 "\n\n## Step 3: Use the formula for the sum of an arithmetic series"
                 "\nWe can use the formula for the sum of an arithmetic series to find the sum: $- (2 \\cdot 50 + (2 + 100) \\cdot \\frac{49}{2})$."
                 "\n\n## Step 4: Simplify the expression"
                 "\nNow, we simplify the expression: $- (100 + 51 \\cdot 49)$."
                 "\n\n## Step 5: Calculate the final answer"
                 "\nFinally, we calculate the final answer: $- (100 + 2499) = -2599$."
                 "\n\nTherefore, the final answer is: $\\boxed{-2599}$.",
    "step1_res": "<|start_header_id|>assistant<|end_header_id|>"
                 "\n\n## Step 1: Decompose the sequence into pairs of consecutive numbers"
                 "\nThe given sequence can be seen as a series of pairs of consecutive numbers: $(1-2) + (3-4) + (5-6) + \\dots + (99-100)$. This decomposition breaks down the sequence into smaller, more manageable parts."
                 "\n\n## Step 2: Observe the pattern of negative numbers"
                 "\nBy evaluating each pair, we notice that each pair results in a negative number: $-1 + (-1) + (-1) + \\dots + (-1)$. This indicates that we are effectively adding the same negative number multiple times."
                 "\n\n## Step 3: Determine the number of negative terms in the sequence"
                 "\nThere are 50 pairs in the sequence, so we add a negative number 50 times."
                 "\n\n## Step 4: Calculate the sum of the negative numbers"
                 "\nTo find the sum of the 50 negative numbers, we multiply the value of the negative term by the number of terms: $-1 \\cdot 50 = -50$."
                 "\n\n## Step 5: Simplify the expression"
                 "\nIn this case, we don't need to simplify the expression further, as the sum is already a simple negative number."
                 "\n\n## Step 6: Express the final answer in boxed notation"
                 "\nTherefore, the final answer is: $\\boxed{-50}$.",
    "step2_res": "<|start_header_id|>assistant<|end_header_id|>"
                 "\n\nUpon reviewing the two approaches, we can see that they both produce the same result: $\\boxed{-50}$."
                 "\n\nThe first approach involves pairing the numbers and calculating the sum of the negative numbers, while the second approach views the sequence as a series of negative numbers added together. Both methods are valid and result in the same answer."
                 "\n\nThis consistency suggests that the problem was approached from different perspectives, each leading to the same solution. However, Approach 1 involves more steps and calculations, making it slightly more complex. Approach 2, on the other hand, provides a more direct and straightforward method for solving the problem."
                 "\n\nIn general, it's beneficial to have multiple approaches to a problem, as they can provide different insights and help verify the accuracy of the solution.",
    "step3_res": "<|start_header_id|>assistant<|end_header_id|>"
                 "\n\n## Step 1: Evaluate Approach 1"
                 "\nApproach 1 involves pairing the numbers and calculating the sum of the negative numbers. This method is rigorous and straightforward, but it requires calculations and may be more time-consuming."
                 "\n\n## Step 2: Evaluate Approach 2"
                 "\nApproach 2 views the sequence as a series of negative numbers added together. This method is also rigorous and provides a direct solution to the problem."
                 "\n\n## Step 3: Compare the diff_of_n_data and method"
                 "\nBoth approaches use complete and reliable diff_of_n_data. They also align with practical applications, as they demonstrate how to solve a sequence problem using different methods."
                 "\n\n## Step 4: Determine the more reliable result"
                 "\nBased on the analysis, both approaches have strengths and weaknesses. Approach 1 is more intricate, while Approach 2 is more direct. However, since they produce the same result, $\\boxed{-50}$, and both meet the criteria, the result is considered reliable from either approach."
                 "\n\n## Step 5: Conclusion"
                 "\nConsidering both approaches meet the criteria of having rigorous logic, complete and reliable diff_of_n_data, and aligning with practical applications, we can conclude that the results are reliable."
                 "\n\nTherefore, the final answer is: $\\boxed{-50}$."
}



print(Template(step_prompt["step1"]).render(problem=data["problem"], approach_1=data["step0_res"], approach_2=data["step1_res"], comparison=data["step2_res"]))

