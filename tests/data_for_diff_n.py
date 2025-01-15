from jinja2 import Template

data = {
    "step1": "You are given a math problem: {{ problem }}."
             "\n\nAnd the following {{ n_solutions | length }} solution ideas:\n"
             "{% for solution in n_solutions %}"
             "- solution {{ loop.index }}: {{ solution }}\n"
             "{% endfor %}"
             "\nPlease review these ideas and select k (1 < k < n) solutions that are both representative and significantly different from each other. Explain briefly why you chose each solution."
             "\n\nRegardless of the approach, always conclude with:"
             "\nTherefore, the indices of the selected solutions are: $\\boxed{answer}$. I hope you like it."
             "\nWhere [answer] is an indices sequence indicating the indices of the selected solutions in the previous solution list, which is ranging from 1 to n."
             "\nMultiple indices must be separated by a comma and a space. For example: $\\boxed{1, 3, 5}$. Do not concatenate the digits (e.g., do not output $\\boxed{135}$).",
    "step2": "Given the math problem: {{ problem }}."
             "\n\nAnd the {{ k_diff_solutions | length }} representative solution ideas:\n"
             "{% for solution in k_diff_solutions %}"
             "- solution {{ loop.index }}: {{ solution }}\n"
             "{% endfor %}"
             "\n\nPlease compare the final results or outcomes of these k solutions to determine if they are consistent with each other. Provide your reasoning in detail.",
    "step3": "Given the math problem: {{ problem }}."
             "\n\nAnd the {{ k_diff_solutions | length }} representative solution ideas:\n"
             "{% for solution in k_diff_solutions %}"
             "- solution {{ loop.index }}: {{ solution }}\n"
             "{% endfor %}"
             "\nAnd the consistency evaluation from Step 2: {{ consistency_evaluation }}"
             "\n\nIf the solutions do not agree, please determine which solution or solutions are the most reliable, and justify your conclusion."
}

temp = Template(data["step2"])

print(
    temp.render(
        problem="123",
        n_solutions=[
            "## Step 1: Calculate the radius $r$ using the formula $r = \sqrt{x^2 + y^2}$, where $x = 0$ and $y = 3$.$r = \sqrt{0^2 + 3^2} = \sqrt{9} = 3$.",
            "123",
            "456"
        ],
        k_diff_solutions=["13", "22343", "6564"],
        consistency_evaluation="hello!!!!!"
    )
)
