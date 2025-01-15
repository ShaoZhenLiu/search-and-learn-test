import regex

text = [
    """After reviewing the given solutions, I have chosen three representative solutions that are significantly different from each other. The reasons for this selection are:

1. Coverage of different calculation steps: Each of the selected solutions covers a distinct set of steps, ensuring that various aspects of the problem are addressed.
2. Clarity of explanation: The selected solutions provide clear and concise descriptions of the calculations and reasoning.
3. Variety in solution structure: The chosen solutions exhibit different structures, such as using multiple steps in one solution and using a simpler structure in another.

The selected solutions are:

1. Solution 1: This solution is chosen because it clearly follows a step-by-step approach, which is easy to follow and provides a clear explanation for each calculation.
2. Solution 4: I selected solution 4 because it offers a concise and straightforward explanation, covering the necessary calculations in just two steps. This provides an example of a more efficient solution structure.
3. Solution 5: This solution is chosen because it provides a detailed and structured approach, including clear explanations and a verification step. This variety in solution structure makes it a valuable representative.

Therefore, the indices of the selected solutions are: $\\boxed{1,4,5}$. I hope you like it.""",
    """After reviewing the given problem solutions, we can see that all of them provide a valid approach to solving the problem. However, some of them are more concise and direct than others.

To select a representative and diverse set of solutions, we need to consider the different strategies and techniques used in each solution. Here are my selected solutions:

- Solution 1 is a clear and straightforward approach that uses a change of variables to simplify the double sum.
- Solution 4 uses an interesting method of expressing the sum as an integral using the definition of the gamma function. This approach is less well-known and provides a unique perspective on the problem.
- Solution 5 uses a clever change of variables to express the double sum as a single integral, which is then evaluated using Fubini's theorem. This approach is concise and well-explained.

These three solutions provide a good representation of the different approaches used to solve the problem. They are all well-written and easy to follow, and they each provide a unique perspective on the problem.

Therefore, the indices of the selected solutions are: $\\boxed{[1, 4, 5]}$. I hope you like it.""",
    """To provide the best selection of solutions, I will review each option and choose the most representative and distinct solutions.

I considered the following criteria for selection:

1. Completeness: The solution must address all steps required to solve the problem.
2. Correctness: The solution must produce the correct final answer.
3. Clarity: The solution must be easy to understand and follow.
4. Uniqueness: The solution must be distinct from other options.

After reviewing the solutions, I selected the following options:

- Solution 1: This solution is straightforward and follows a clear step-by-step approach. It correctly finds the value of $f(-2)$, $f(-1)$, and $f(0)$ and then adds them up to get the final answer.

- Solution 5: This solution is unique in that it explicitly handles the undefined value of $f(0)$. By assuming $f(0) = 0$ for the sake of calculation, it provides a different perspective on how to approach the problem.

I did not select Solution 2 because it does not provide a clear step-by-step approach to solving the problem and makes an incorrect assumption about the behavior of the function at $x = -2$ and $x = 0$.

I did not select Solution 3 and Solution 4 because they are essentially identical to Solution 1 and do not provide any additional insights or perspectives.

Therefore, the indices of the selected solutions are: $\\boxed{[1, 5]}$. I hope you like it.""",
    """After reviewing the solution ideas, I have chosen the top 3 solutions that are both representative and significantly different from each other. 

I have selected solutions 1, 2, and 3 because they vary in the way they phrase and calculate the prime factorization and the number of divisors. 

- Solution 1 provides a clear, step-by-step approach, explicitly stating the formula for finding the number of divisors.
- Solution 2 offers a slightly different phrasing of the prime factorization and the formula, but the calculation remains the same.
- Solution 3 uses a more general rule for finding the number of divisors and explicitly mentions the exponent of 7 as 0, showing an alternative way to think about the problem.

These solutions cover different aspects of the problem and provide readers with various approaches to solve it.

Therefore, the indices of the selected solutions are: $\\boxed{1, 2, 3}$. I hope you like it.""",
    """I have reviewed the given solution ideas for the math problem. After analysis, I recommend selecting solutions 1, 2, and 4 as they are representative and significantly different from each other. 

I chose solution 1 because it provides a clear and step-by-step approach to calculating the average speed for each student, including the calculation of the average speeds.

I chose solution 2 because it uses a more direct and efficient method to calculate the distances and times for each student, resulting in a simpler calculation of the average speeds.

I chose solution 4 because it introduces the concept of using the distance formula to find the total distance traveled by each student and provides a clear explanation of the method used.

The other solutions (3 and 5) are less representative and significantly different from each other as they either have minor deviations in calculation steps or have unnecessary complexities, making them less comparable to the selected solutions.

Therefore, the indices of the selected solutions are: $\\boxed{[1, 2, 4]}$. I hope you like it.""",
]
# 使用递归引用组1来处理任意深度的嵌套
pattern = r'\\boxed{((?:[^{}]+|{(?1)})*)}'
for t in text:
    match = regex.search(pattern, t)
    if match:
        # print(match.group(1))  # 应输出: \frac{1}{2}
        print(regex.findall(r"\d+", match.group(1)))
    else:
        print("未找到匹配内容。")
